use core::panic;
use crossbeam::channel;
use rand::{rngs::ThreadRng, Rng};
use serde_json::Value;
use tokenizers::Tokenizer;
use tracing::{instrument, Level};

use std::fs;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

#[derive(Clone)]
struct TokenSeparater {
    pub bos_token: (u32, String),
    pub eos_token: (u32, String),
    pub pad_token: (u32, String),
}

impl TokenSeparater {
    fn parse_token_id(token_value: &Value) -> Option<u32> {
        token_value
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .or_else(|| token_value.as_str()?.parse::<u32>().ok())
    }

    fn find_added_token(config_json: &Value, content: &str) -> Option<(u32, String)> {
        let decoder = config_json.get("added_tokens_decoder")?.as_object()?;
        decoder.iter().find_map(|(id, token)| {
            let token_content = token.get("content")?.as_str()?;
            if token_content == content {
                Some((id.parse::<u32>().ok()?, token_content.to_owned()))
            } else {
                None
            }
        })
    }

    fn find_special_token(config_json: &Value, field: &str) -> Option<(u32, String)> {
        let token_content = config_json.get(field)?.as_str()?;
        Self::find_added_token(config_json, token_content)
    }

    fn find_special_token_id(config_json: &Value, field: &str) -> Option<u32> {
        Self::parse_token_id(config_json.get(field)?)
    }

    fn qwen_tokens(config_json: &Value) -> Option<Self> {
        let pad_token = Self::find_special_token(config_json, "pad_token")
            .or_else(|| Self::find_added_token(config_json, "<|endoftext|>"))
            .or_else(|| {
                Self::find_special_token_id(config_json, "pad_token_id")
                    .map(|id| (id, "<|endoftext|>".to_owned()))
            })?;

        let bos_token = Self::find_special_token(config_json, "bos_token")
            .or_else(|| Self::find_added_token(config_json, "<|im_start|>"))
            .or_else(|| {
                Self::find_special_token_id(config_json, "bos_token_id")
                    .map(|id| (id, "<|im_start|>".to_owned()))
            })?;

        let eos_token = Self::find_special_token(config_json, "eos_token")
            .or_else(|| Self::find_added_token(config_json, "<|im_end|>"))
            .or_else(|| {
                Self::find_special_token_id(config_json, "eos_token_id")
                    .map(|id| (id, "<|im_end|>".to_owned()))
            })?;

        Some(Self {
            bos_token,
            eos_token,
            pad_token,
        })
    }

    fn mistral_tokens(config_json: &Value) -> Option<Self> {
        // Prefer canonical Mistral sentence boundary tokens over chat-template wrappers.
        // Some instruct checkpoints expose `<s>[INST]` / `[/INST]` in tokenizer fields,
        // but those are conversation delimiters instead of true BOS/EOS boundaries.
        let bos_token = Self::find_added_token(config_json, "<s>")
            .or_else(|| {
                let token = Self::find_special_token(config_json, "bos_token")?;
                if token.1 == "<s>" {
                    Some(token)
                } else {
                    None
                }
            })
            .or_else(|| Self::find_added_token(config_json, "<s>"))
            .or_else(|| Self::find_special_token_id(config_json, "bos_token_id").map(|id| (id, "<s>".to_owned())))
            .or_else(|| Self::find_special_token(config_json, "bos_token"))
            .unwrap_or_else(|| (1, "<s>".to_owned()));

        let eos_token = Self::find_added_token(config_json, "</s>")
            .or_else(|| {
                let token = Self::find_special_token(config_json, "eos_token")?;
                if token.1 == "</s>" {
                    Some(token)
                } else {
                    None
                }
            })
            .or_else(|| Self::find_added_token(config_json, "</s>"))
            .or_else(|| Self::find_special_token_id(config_json, "eos_token_id").map(|id| (id, "</s>".to_owned())))
            .or_else(|| Self::find_special_token(config_json, "eos_token"))
            .unwrap_or_else(|| (2, "</s>".to_owned()));

        let pad_token = Self::find_special_token(config_json, "pad_token")
            .or_else(|| Self::find_added_token(config_json, "<pad>"))
            .or_else(|| Self::find_special_token_id(config_json, "pad_token_id").map(|id| (id, "<pad>".to_owned())))
            .unwrap_or_else(|| eos_token.clone());

        Some(Self {
            bos_token,
            eos_token,
            pad_token,
        })
    }

    pub fn new(path: &str) -> Self {
        let data = fs::read_to_string(path).expect("Failed to read tokenizer config file");
        let config_json: Value =
            serde_json::from_str(&data).expect("Failed to parse tokenizer config file as JSON");

        if let Some(class) = config_json.get("tokenizer_class") {
            if matches!(
                class.as_str(),
                Some("Qwen2Tokenizer")
                    | Some("Qwen2TokenizerFast")
                    | Some("Qwen3Tokenizer")
                    | Some("Qwen3TokenizerFast")
            ) {
                if let Some(tokens) = Self::qwen_tokens(&config_json) {
                    return tokens;
                }

                unimplemented!("Qwen tokenizer config is missing required special tokens");
            } else if class.as_str() == Some("PreTrainedTokenizerFast") {
                let bos_token = Self::find_added_token(&config_json, "<|begin_of_text|>")
                    .expect("Llama tokenizer missing <|begin_of_text|> token");
                let eos_token = Self::find_added_token(&config_json, "<|end_of_text|>")
                    .expect("Llama tokenizer missing <|end_of_text|> token");
                let pad_token = Self::find_added_token(&config_json, "<|finetune_right_pad_id|>")
                    .unwrap_or_else(|| eos_token.clone());

                return TokenSeparater {
                    bos_token,
                    eos_token,
                    pad_token,
                };
            } else if matches!(
                class.as_str(),
                Some("MistralCommonTokenizer")
                    | Some("Tekkenizer")
                    | Some("TekkenTokenizer")
            ) {
                if let Some(tokens) = Self::mistral_tokens(&config_json) {
                    return tokens;
                }
            } else {
                // Mistral tokenizer_class can vary across checkpoints and serving stacks.
                // Fall back to token-content based detection before failing hard.
                if let Some(tokens) = Self::mistral_tokens(&config_json) {
                    return tokens;
                }
            }
            unimplemented!(
                "Currently support Qwen2/Qwen2.5/Qwen3, Llama-3.1 and Mistral-Small-24B-Instruct-2501 class model"
            );
        } else {
            if let Some(tokens) = Self::mistral_tokens(&config_json) {
                return tokens;
            }
            unimplemented!(
                "Currently support Qwen2/Qwen2.5/Qwen3, Llama-3.1 and Mistral-Small-24B-Instruct-2501 class model"
            );
        }
    }
}

/// TokenSampler: asynchronous sampling and caching mechanism for random text block generation
pub struct TokenSampler {
    tokenizer: Tokenizer,
    block_size: usize,
    fst_rx: channel::Receiver<String>,
    snd_cmd_tx: channel::Sender<usize>,
    snd_data_rxs: Vec<channel::Receiver<String>>,
}

struct BatchSampleContext {
    pub tokens: Vec<u32>,
    pub begin: usize,
    batch_size: usize,
    rng: ThreadRng,
}

impl BatchSampleContext {
    pub fn new(tokenizer: &Tokenizer, batch_size: usize) -> Self {
        let tokens = Vec::with_capacity(batch_size);
        let begin = 0;
        let rng = rand::thread_rng();

        let mut ctx = Self {
            tokens,
            begin,
            batch_size,
            rng,
        };
        ctx.reload(tokenizer);

        ctx
    }
    pub fn reload(&mut self, tokenizer: &Tokenizer) {
        // tracing::info!("BatchSampler reload.");
        // Continuously reload Batchsampler
        let vocab_size = tokenizer.get_vocab_size(true) as u32;

        let tokens: Vec<u32> = (0..self.batch_size)
            .map(|_| self.rng.gen_range(0..vocab_size))
            .collect();
        let dec = tokenizer.decode(&tokens, false).unwrap_or_default();
        let enc = tokenizer.encode(dec, false).unwrap();
        self.tokens = enc.get_ids().to_owned();
        self.begin = 0;
    }
}

enum BatchSampleError {
    InvalidLength(String, usize),
    EndOfContext,
}

impl TokenSampler {
    /// init TokenSampler
    pub fn new(
        tokenizer: Tokenizer,
        tokenizer_config_path: String,
        num_producers: usize,
        channel_capacity: usize,
        block_size: usize,
    ) -> Self {
        let sep = TokenSeparater::new(&tokenizer_config_path);

        // Primary (fst) channel for full blocks
        // 128K blocks => 1M tokens => ~10MB memory
        let primary_channel_cap = 1024 * 1024 / block_size;
        let (fst_tx, fst_rx) = channel::bounded::<String>(primary_channel_cap);

        // Secondary (snd) channel for incomplete blocks
        let (snd_cmd_tx, snd_rx) = channel::unbounded::<usize>();
        let (snd_data_txs, snd_data_rxs) = (0..block_size)
            .map(|_| channel::bounded(channel_capacity * 2))
            .collect::<(Vec<_>, Vec<_>)>();
        let snd_data_txs = Arc::new(snd_data_txs);

        // init producer threads
        for i in 0..num_producers {
            let tokenizer0 = tokenizer.clone();
            let sep = sep.clone();
            let fst_data_tx = fst_tx.clone();
            let snd_cmd_rx = snd_rx.clone();
            let snd_data_txs = snd_data_txs.clone();

            thread::spawn(move || {
                Self::producer_loop_v2(
                    i,
                    tokenizer0,
                    sep,
                    block_size,
                    channel_capacity,
                    fst_data_tx,
                    snd_cmd_rx,
                    snd_data_txs,
                );
            });
        }

        tracing::info!("Warmup start...");
        for size in 1..block_size {
            for _ in 0..channel_capacity {
                let _ = snd_cmd_tx.send(size);
            }
        }
        tracing::info!("Warmup finished!");

        Self {
            tokenizer,
            block_size,
            fst_rx,
            snd_cmd_tx,
            snd_data_rxs,
        }
    }

    #[allow(unused)]
    #[deprecated]
    fn producer_loop(
        id: usize,
        tokenizer: Tokenizer,
        splitter: Vec<String>,
        block_size: usize,
        channel_capacity: usize,
        fst_data_tx: channel::Sender<String>,
        snd_cmd_rx: channel::Receiver<usize>,
        snd_data_txs: Arc<Vec<channel::Sender<String>>>,
    ) {
        let mut local_samples = Vec::new();
        loop {
            // Make up consumed non-complete block
            match snd_cmd_rx.try_recv() {
                Ok(size) => {
                    // received messages -> generate corresponding sample and send to ragged channel
                    let ragged_tx = snd_data_txs.get(size - 1).unwrap();
                    if ragged_tx.len() < channel_capacity {
                        let ragged_sample = Self::generate_block(&tokenizer, &splitter, size);
                        let _ = ragged_tx.try_send(ragged_sample);
                    }
                }
                Err(channel::TryRecvError::Empty) => {}
                Err(channel::TryRecvError::Disconnected) => {
                    tracing::debug!("Producer-{id} snd_rx disconnected, exiting");
                    break;
                }
            }

            // Refill new complete block
            let new_sample = if local_samples.is_empty() {
                Self::generate_block(&tokenizer, &splitter, block_size)
            } else {
                local_samples.pop().unwrap()
            };

            // Try to send complete block
            match fst_data_tx.try_send(new_sample) {
                Ok(_) => continue, // Refill successfully, next round
                Err(channel::TrySendError::Full(x)) => {
                    local_samples.push(x);
                    // Primary channel is full, waiting for secondary channel
                    match snd_cmd_rx.recv_timeout(Duration::from_millis(10)) {
                        Ok(size) => {
                            // Make up consumed non-complete block
                            let ragged_tx = snd_data_txs.get(size - 1).unwrap();
                            if !ragged_tx.is_full() {
                                let ragged_sample =
                                    Self::generate_block(&tokenizer, &splitter, size);
                                let _ = ragged_tx.try_send(ragged_sample);
                            }
                        }
                        Err(channel::RecvTimeoutError::Timeout) => {
                            // timeout -> do nothing, continue next round
                            continue;
                        }
                        Err(channel::RecvTimeoutError::Disconnected) => {
                            tracing::debug!("Producer-{id} notify_rx disconnected, exiting");
                            break;
                        }
                    }
                }
                Err(channel::TrySendError::Disconnected(_)) => {
                    // receiver closed -> exit
                    tracing::debug!("Producer-{id} tx disconnected, exiting");
                    break;
                }
            }
        }
        tracing::debug!("Producer-{id} exited");
    }

    fn generate_block(tokenizer: &Tokenizer, splitter: &[String], n: usize) -> String {
        let mut rng = rand::thread_rng();
        let vocab_size = tokenizer.get_vocab_size(true) as u32;

        // let generate_time = std::time::Instant::now();
        match n {
            0 => return String::new(),
            1 => return splitter[0].clone(),
            2 => {
                return if splitter.len() == 2 {
                    format!("{}{}", splitter[0], splitter[1])
                } else {
                    splitter[0].repeat(2)
                };
            }
            _ => {}
        }

        loop {
            let tokens: Vec<u32> = (0..2 * n).map(|_| rng.gen_range(0..vocab_size)).collect();
            let decoded = tokenizer.decode(&tokens, false).unwrap_or_default();
            let encoded = tokenizer.encode(decoded, false).unwrap();
            let mut ids = encoded.get_ids().to_vec();
            ids.truncate(n.saturating_sub(2));

            let mut result = tokenizer.decode(&ids, false).unwrap_or_default();

            // add splitter
            if splitter.len() == 2 {
                result.insert_str(0, &splitter[0]);
                result.push_str(&splitter[1]);
            } else {
                result.insert_str(0, &splitter[0]);
                result.push_str(&splitter[0]);
            }

            // verify the length
            let reencoded_len = tokenizer
                .encode(result.clone(), false)
                .unwrap()
                .get_ids()
                .len();
            if reencoded_len == n {
                // let duration = generate_time.elapsed();
                // tracing::info!("Generated block of size {n} in {duration:?}");
                return result;
            }
        }
    }

    fn producer_loop_v2(
        id: usize,
        tokenizer: Tokenizer,
        sep: TokenSeparater,
        block_size: usize,
        channel_capacity: usize,
        fst_data_tx: channel::Sender<String>,
        snd_cmd_rx: channel::Receiver<usize>,
        snd_data_txs: Arc<Vec<channel::Sender<String>>>,
    ) {
        let mut ctx = BatchSampleContext::new(&tokenizer, 2048);
        let mut local_samples = Vec::new();
        loop {
            // Make up consumed non-complete block
            match snd_cmd_rx.try_recv() {
                Ok(size) => {
                    // received messages -> generate corresponding sample and send to ragged channel
                    let ragged_tx = snd_data_txs.get(size - 1).unwrap();
                    if ragged_tx.len() < channel_capacity {
                        let ragged_sample = Self::generate_block_v2(
                            &tokenizer,
                            &sep,
                            snd_data_txs.as_ref(),
                            size,
                            &mut ctx,
                        );
                        let _ = ragged_tx.try_send(ragged_sample);
                    }
                }
                Err(channel::TryRecvError::Empty) => {}
                Err(channel::TryRecvError::Disconnected) => {
                    tracing::debug!("Producer-{id} snd_rx disconnected, exiting");
                    break;
                }
            }

            // Refill new complete block
            let new_sample = if local_samples.is_empty() {
                Self::generate_block_v2(
                    &tokenizer,
                    &sep,
                    snd_data_txs.as_ref(),
                    block_size,
                    &mut ctx,
                )
            } else {
                local_samples.pop().unwrap()
            };

            // Try to send complete block
            match fst_data_tx.try_send(new_sample) {
                Ok(_) => continue, // Refill successfully, next round
                Err(channel::TrySendError::Full(x)) => {
                    local_samples.push(x);
                    // Primary channel is full, waiting for secondary channel
                    match snd_cmd_rx.recv_timeout(Duration::from_millis(5)) {
                        Ok(size) => {
                            // Make up consumed non-complete block
                            let ragged_tx = snd_data_txs.get(size - 1).unwrap();
                            if !ragged_tx.is_full() {
                                let ragged_sample = Self::generate_block_v2(
                                    &tokenizer,
                                    &sep,
                                    snd_data_txs.as_ref(),
                                    size,
                                    &mut ctx,
                                );
                                let _ = ragged_tx.try_send(ragged_sample);
                            }
                        }
                        Err(channel::RecvTimeoutError::Timeout) => {
                            continue;
                        }
                        Err(channel::RecvTimeoutError::Disconnected) => {
                            tracing::debug!("Producer-{id} notify_rx disconnected, exiting");
                            break;
                        }
                    }
                }
                Err(channel::TrySendError::Disconnected(_)) => {
                    // receiver closed -> exit
                    tracing::debug!("Producer-{id} tx disconnected, exiting");
                    break;
                }
            }
        }
        tracing::debug!("Producer-{id} exited");
    }

    fn generate_block_v2(
        tokenizer: &Tokenizer,
        sep: &TokenSeparater,
        snd_data_txs: &[channel::Sender<String>],
        n: usize,
        ctx: &mut BatchSampleContext,
    ) -> String {
        tracing::trace!("generate block v2: n={n}");
        let mut result = None;
        while result.is_none() {
            match Self::generate_block_v2_inner(tokenizer, sep, n, ctx) {
                Ok(sample) => {
                    result = Some(sample);
                }
                Err(BatchSampleError::InvalidLength(inv_sample, size)) => {
                    if let Some(ragged_tx) = snd_data_txs.get(size - 1) {
                        let _ = ragged_tx.try_send(inv_sample);
                    } else {
                        tracing::debug!("Expect length: {n}, get length: {size}");
                    }
                }
                Err(BatchSampleError::EndOfContext) => {
                    ctx.reload(tokenizer);
                }
            }
        }
        result.unwrap()
    }

    fn generate_block_v2_inner(
        tokenizer: &Tokenizer,
        sep: &TokenSeparater,
        n: usize,
        ctx: &mut BatchSampleContext,
    ) -> Result<String, BatchSampleError> {
        match n {
            0 => return Ok(String::new()),
            1 => return Ok(sep.pad_token.1.clone()),
            2 => {
                return Ok(format!("{}{}", &sep.bos_token.1, &sep.eos_token.1));
            }
            _ => {}
        }

        let batch_tokens = &mut ctx.tokens;
        let begin = ctx.begin;
        let mut end = begin + n - 2;

        while end <= batch_tokens.len() {
            let mut miss = 0;
            loop {
                let tokens = &batch_tokens[begin..end];
                let mut asm_tokens = Vec::with_capacity(n + miss);
                asm_tokens.push(sep.bos_token.0);
                asm_tokens.extend_from_slice(tokens);
                asm_tokens.push(sep.eos_token.0);
                let result = tokenizer.decode(&asm_tokens, false).unwrap_or_default();
                let validate_len = tokenizer
                    .encode(result.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                if validate_len < n {
                    miss += 1;
                    end += 1;
                    if end > batch_tokens.len() {
                        return Err(BatchSampleError::EndOfContext);
                    }
                } else if validate_len > n {
                    // NOTE: non-unit stepping
                    ctx.begin = end;
                    return Err(BatchSampleError::InvalidLength(result, validate_len));
                } else {
                    // Valid block found!
                    ctx.begin = end;
                    return Ok(result);
                }
            }
        }
        Err(BatchSampleError::EndOfContext)
    }

    /// Public client interface
    #[instrument(skip_all, fields(block_size = n), target = "inflate::inner" level = Level::DEBUG)]
    pub fn gen_string(&self, n: usize) -> String {
        if self.block_size == n {
            if let Ok(sample) = self.fst_rx.recv() {
                return sample;
            }
        }

        if let Some(rx) = self.snd_data_rxs.get(n - 1) {
            if let Ok(sample) = rx.try_recv() {
                self.snd_cmd_tx.send(n).unwrap();
                return sample;
            } else {
                self.snd_cmd_tx.send(n).unwrap();
                return rx.recv().unwrap();
            }
        } else {
            panic!("No channel for incomplete block size {n}");
        }
    }

    pub fn get_tokenizer(&self) -> Tokenizer {
        self.tokenizer.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::Instant;
    use std::time::{SystemTime, UNIX_EPOCH};

    const QWEN2_EOS_TOKEN: u32 = 151645;
    const QWEN2_BOS_TOKEN: u32 = 151644;
    #[allow(unused)]
    const QWEM2_PAD_TOKEN: u32 = 151643;

    fn validate_block_generation(
        tokenizer: &Tokenizer,
        size: usize,
        stride: usize,
    ) -> (String, usize) {
        let mut rng = rand::thread_rng();
        let vocab_size = tokenizer.get_vocab_size(true) as u32;
        let tokens: Vec<u32> = (0..2 * size)
            .map(|_| rng.gen_range(0..vocab_size))
            .collect();
        let decoded = tokenizer.decode(&tokens, false).unwrap_or_default();
        let encoded = tokenizer.encode(decoded, false).unwrap();
        let mut all_tokens = encoded.get_ids().to_vec();
        all_tokens.truncate(size);

        let mut ret = String::with_capacity(size);
        let mut cnt = 0;
        let mut begin = 0;
        let mut end = begin + stride - 2;

        while end <= all_tokens.len() {
            let mut miss = 0;
            loop {
                let tokens = &all_tokens[begin..end];
                let mut asm_tokens = Vec::with_capacity(stride + miss);
                asm_tokens.push(QWEN2_BOS_TOKEN);
                asm_tokens.extend_from_slice(tokens);
                asm_tokens.push(QWEN2_EOS_TOKEN);
                let result = tokenizer.decode(&asm_tokens, false).unwrap_or_default();
                let reencoded_len = tokenizer
                    .encode(result.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                if reencoded_len < stride {
                    miss += 1;
                    end += 1;
                    if end > all_tokens.len() {
                        break;
                    }
                } else if reencoded_len > stride {
                    // NOTE: non-unit stepping
                    begin = end;
                    end += stride - 2;
                    break;
                } else {
                    begin = end;
                    end += stride - 2;
                    cnt += 1;
                    ret.push_str(&result);
                    break;
                }
            }
        }
        (ret, cnt)
    }

    /// test encode/decode latency increases with the number of tokens n.
    ///
    /// output format:
    /// ```
    /// n=16, time=1.23ms
    /// n=32, time=2.12ms
    /// ...
    /// ```
    #[test]
    fn test_gen_string_latency_scaling() {
        // init tokenizer
        let tokenizer_path = "data/tokenizer.json"; // your own path
        let tokenizer = Tokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer");

        println!("==== TokenSampler decode latency test ====");
        println!("{:<8} | {:<12}", "n", "time (ms)");
        println!("--------------------------------");

        let mut total_cnt = 0;
        let mut total_elapsed = 0.;
        let stride = 16;
        for _ in 0..16 {
            let start = Instant::now();
            for _ in 0..5 {
                // let _ = generate_block(&tokenizer, n);
                let (result, cnt) = validate_block_generation(&tokenizer, 2048, stride);
                let elapsed = start.elapsed();
                let elapsed_ms = (elapsed.as_secs_f64() * 1000.0 * 100.0).round() / 100.0; // keep two decimal places
                let reencoded_len = tokenizer
                    .encode(result.clone(), false)
                    .unwrap()
                    .get_ids()
                    .len();
                let s = if reencoded_len == cnt * 16 {
                    "OK"
                } else {
                    println!("encode len: {reencoded_len} | expected: {}", cnt * 16);
                    "Err"
                };
                println!("{s}:> {:<8} | {:<12.2}", cnt, elapsed_ms);
                total_cnt += cnt;
                total_elapsed += elapsed_ms;
            }
        }
        println!("--------------------------------");
        println!(
            "Speed: {:<4}ms/block | block size: {stride}",
            total_elapsed / total_cnt as f64
        );
    }

    fn write_temp_tokenizer_config(contents: &str) -> PathBuf {
        let filename = format!(
            "tokenizer_config_{}_{}.json",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(filename);
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn test_token_separator_supports_llama31_instruct() {
        let config = r#"{
            "tokenizer_class": "PreTrainedTokenizerFast",
            "added_tokens_decoder": {
                "128000": {"content": "<|begin_of_text|>"},
                "128001": {"content": "<|end_of_text|>"},
                "128004": {"content": "<|finetune_right_pad_id|>"}
            }
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.bos_token, (128000, "<|begin_of_text|>".to_owned()));
        assert_eq!(sep.eos_token, (128001, "<|end_of_text|>".to_owned()));
        assert_eq!(
            sep.pad_token,
            (128004, "<|finetune_right_pad_id|>".to_owned())
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_token_separator_llama_fallback_pad_to_eos() {
        let config = r#"{
            "tokenizer_class": "PreTrainedTokenizerFast",
            "added_tokens_decoder": {
                "128000": {"content": "<|begin_of_text|>"},
                "128001": {"content": "<|end_of_text|>"}
            }
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.pad_token, sep.eos_token);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_token_separator_supports_mistral_small_24b_instruct_2501() {
        let config = r#"{
            "tokenizer_class": "MistralCommonTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>",
            "added_tokens_decoder": {
                "1": {"content": "<s>"},
                "2": {"content": "</s>"}
            }
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.bos_token, (1, "<s>".to_owned()));
        assert_eq!(sep.eos_token, (2, "</s>".to_owned()));
        assert_eq!(sep.pad_token, sep.eos_token);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_token_separator_supports_qwen3_14b() {
        let config = r#"{
            "tokenizer_class": "Qwen3Tokenizer",
            "bos_token": "<|im_start|>",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "added_tokens_decoder": {
                "151643": {"content": "<|endoftext|>"},
                "151644": {"content": "<|im_start|>"},
                "151645": {"content": "<|im_end|>"}
            }
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.bos_token, (151644, "<|im_start|>".to_owned()));
        assert_eq!(sep.eos_token, (151645, "<|im_end|>".to_owned()));
        assert_eq!(sep.pad_token, (151643, "<|endoftext|>".to_owned()));

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_token_separator_mistral_fallback_without_added_tokens_decoder() {
        let config = r#"{
            "tokenizer_class": "MistralCommonTokenizer",
            "bos_token": "<s>",
            "eos_token": "</s>"
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.bos_token, (1, "<s>".to_owned()));
        assert_eq!(sep.eos_token, (2, "</s>".to_owned()));
        assert_eq!(sep.pad_token, sep.eos_token);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn test_token_separator_mistral_prefers_canonical_sentence_boundaries() {
        let config = r#"{
            "tokenizer_class": "Tekkenizer",
            "bos_token": "<s>[INST]",
            "eos_token": "[/INST]",
            "added_tokens_decoder": {
                "1": {"content": "<s>"},
                "2": {"content": "</s>"},
                "3": {"content": "<s>[INST]"},
                "4": {"content": "[/INST]"}
            }
        }"#;

        let path = write_temp_tokenizer_config(config);
        let sep = TokenSeparater::new(path.to_str().unwrap());

        assert_eq!(sep.bos_token, (1, "<s>".to_owned()));
        assert_eq!(sep.eos_token, (2, "</s>".to_owned()));

        fs::remove_file(path).unwrap();
    }
}
