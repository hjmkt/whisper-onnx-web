import * as ort from "onnxruntime-web";
import { get_tokenizer } from "./tokenizer";
import pako from "pako";

function DecodingResult(
    audio_features,
    language,
    language_probs,
    tokens,
    text,
    avg_logprob,
    no_speech_prob,
    temperature,
    compression_ratio
) {
    this.audio_features = audio_features;
    this.language = language;
    this.language_probs = language_probs;
    this.tokens = tokens;
    this.text = text;
    this.avg_logprob = avg_logprob;
    this.no_speech_prob = no_speech_prob;
    this.temperature = temperature;
    this.compression_ratio = compression_ratio;
}

export function Tensor(shape, dtype, data) {
    this.shape = shape;
    if (data === null || data === undefined) {
        if (dtype === undefined) {
            dtype = "float32";
        }
        if (dtype == "float32") {
            this.data = new Float32Array(shape.reduce((a, b) => a * b, 1));
        } else if (dtype == "int32") {
            this.data = new Int32Array(shape.reduce((a, b) => a * b, 1));
        } else {
            console.assert(false, "Unsupported dtype");
        }
    } else {
        this.data = data;
    }
    this.dtype = dtype;

    this.sliceStep = function (step) {
        let newShape = [Math.floor(this.shape[0] / step), this.shape[1]];
        let newTensor = new Tensor(newShape, this.dtype);
        for (let i = 0; i < newShape[0]; i++) {
            for (let j = 0; j < newShape[1]; j++) {
                newTensor.data[i * newShape[1] + j] =
                    this.data[i * step * newShape[1] + j];
            }
        }
        return newTensor;
    };

    this.slice = function (...args) {
        let newTensor = new Tensor(this.shape, this.dtype);
        if (args[0] === undefined) {
            if (newTensor.shape[0] > args[1]) {
                newTensor.shape[0] = args[1];
                let stride = this.shape.slice(1).reduce((a, b) => a * b, 1);
                newTensor.data = this.data.slice(0, stride * args[1]);
            }
        } else if (args[1] === undefined) {
            if (args[0] > this.shape[0]) {
                console.assert(false, "Slice index out of range");
            } else {
                newTensor.shape[0] = this.shape[0] - args[0];
                let stride = this.shape.slice(1).reduce((a, b) => a * b, 1);
                newTensor.data = this.data.slice(args[0] * stride);
            }
        } else {
            if (args[0] >= args[1]) {
                console.assert(false, "Slice index out of range");
            } else if (args[0] > this.shape[0] || args[1] > this.shape[0]) {
                console.assert(false, "Slice index out of range");
            } else {
                newTensor.shape[0] = args[1] - args[0];
                let stride = this.shape.slice(1).reduce((a, b) => a * b, 1);
                newTensor.data = this.data.slice(
                    args[0] * stride,
                    args[1] * stride
                );
            }
        }
        return newTensor;
    };

    this.slice2d = function (start, end) {
        if (end == undefined) {
            end = this.shape[1];
        }
        if (start < 0) {
            start += this.shape[1];
        }
        if (end < 0) {
            end += this.shape[1];
        }
        let newShape = [this.shape[0], end - start];
        let newTensor = new Tensor(newShape, this.dtype);
        for (let i = 0; i < newShape[0]; i++) {
            for (let j = 0; j < newShape[1]; j++) {
                newTensor.data[i * newShape[1] + j] =
                    this.data[i * this.shape[1] + j + start];
            }
        }
        return newTensor;
    };

    this.size = function () {
        return this.shape.reduce((a, b) => a * b, 1);
    };

    this.softmax2d = function () {
        let newTensor = new Tensor(this.shape, this.dtype);
        for (let i = 0; i < this.shape[0]; i++) {
            let slice = this.slice2d(
                i * this.shape[1],
                (i + 1) * this.shape[1]
            );
            let smax = softmax(slice.data);
            for (let j = 0; j < smax.length; j++) {
                newTensor.data[i * this.shape[1] + j] = smax[j];
            }
        }
        return newTensor;
    };

    this.log_softmax2d = function () {
        let newTensor = new Tensor(this.shape, this.dtype);
        for (let i = 0; i < this.shape[0]; i++) {
            let slice = this.slice2d(
                i * this.shape[1],
                (i + 1) * this.shape[1]
            );
            let smax = softmax(slice.data);
            for (let j = 0; j < smax.length; j++) {
                newTensor.data[i * this.shape[1] + j] = Math.log(smax[j]);
            }
        }
        return newTensor;
    };

    // for 1d tensor
    this.pad = function (pad) {
        let newShape = [this.shape[0] + pad];
        let newTensor = new Tensor(newShape, this.dtype);
        for (let i = 0; i < this.shape[0]; i++) {
            newTensor.data[i] = this.data[i];
        }
        return newTensor;
    };

    this.reshape = function (shape) {
        let newTensor = new Tensor(shape, this.dtype);
        if (this.size() != shape.reduce((a, b) => a * b, 1)) {
            console.assert(false, "Reshape size mismatch");
        }
        newTensor.shape = shape;
        newTensor.data.set(this.data);
        return newTensor;
    };
}

function compressionRatio(text) {
    const textBytes = new TextEncoder().encode(text);
    const compressedBytes = pako.deflate(textBytes);
    return textBytes.length / compressedBytes.length;
}

function softmax(arr) {
    const expArr = arr.map((x) => Math.exp(x));
    const sum = expArr.reduce(
        (accumulator, currentValue) => accumulator + currentValue,
        0
    );

    return expArr.map((x) => x / sum);
}

export function DecodingOptions() {
    this.task = "transcribe";
    this.language = null;
    this.temperature = 0.0;
    this.sample_len = null;
    this.best_of = null;
    this.beam_size = null;
    this.patience = null;
    this.length_penalty = null;
    this.prompt = null;
    this.prefix = null;
    this.suppress_tokens = "-1";
    this.suppress_blank = true;
    this.without_timestamps = false;
    this.max_initial_timestamp = 1.0;
    this.fp16 = false;
}

function max(array) {
    return array.reduce((a, b) => (a > b ? a : b));
}

function Inference(model, initial_token_length) {
    this.model = model;
    this.initial_token_length = initial_token_length;

    this.logits = async function (tokens) {
        if (tokens.shape.slice(-1)[0] > this.initial_token_length) {
            tokens = tokens.slice2d(-1);
        }
        let offset = this.model.self_attn_value_cache.dims[1] - 1;
        let positional_embedding = (
            await this.model.positional_embedding
        ).slice(offset, offset + tokens.shape[1]);
        let [l, new_self_attn_key_cache, new_self_attn_value_cache] =
            await this.model.decoder(
                tokens,
                this.model.self_attn_key_cache,
                this.model.self_attn_value_cache,
                this.model.cross_attn_key_cache,
                this.model.cross_attn_value_cache,
                positional_embedding
            );
        this.model.self_attn_key_cache = new_self_attn_key_cache;
        this.model.self_attn_value_cache = new_self_attn_value_cache;
        return l;
    };
}

function MaximumLikelihoodRanker(length_penalty) {
    this.length_penalty = length_penalty;

    this.rank = function (tokens, sum_logprobs) {
        function scores(logprobs, lengths) {
            let result = [];
            for (let i = 0; i < logprobs.length; i++) {
                let logprob = logprobs[i];
                let length = lengths[i];
                let penalty = null;
                if (this.length_penalty == null) {
                    penalty = length;
                } else {
                    penalty = Math.pow((5 + length) / 6, this.length_penalty);
                }
                result.push(logprob / penalty);
            }
            return result;
        }

        //let lengths = tokens.map((s) => s.map((t) => t.size()));
        //let lengths = tokens.map((s) => s.length);
        //let maxIndices = sum_logprobs.data.map((p, i) => scores(p, lengths[i]).indexOf(max(scores(p, lengths[i]))));
        let maxIndices = [0];
        return maxIndices;
    };
}

function GreedyDecoder(temperature, eot) {
    this.temperature = temperature;
    this.eot = eot;

    this.update = function (tokens, logits, sum_logprobs) {
        let next_tokens = null;
        next_tokens = [logits.data.indexOf(max(logits.data))];

        let logprobs = logits.log_softmax2d();
        let current_logprobs = [logprobs.data[next_tokens[0]]];
        if (tokens.data.slice(-1)[0] != this.eot) {
            sum_logprobs[0] += current_logprobs.slice(-1)[0];
        }

        if (tokens.data.slice(-1)[0] == this.eot) {
            next_tokens[0] = this.eot;
        }

        let new_tokens = [];
        for (let i = 0; i < tokens.size(); i++) {
            new_tokens.push(tokens.data[i]);
        }
        for (let i = 0; i < next_tokens.length; i++) {
            new_tokens.push(next_tokens[i]);
        }
        tokens = new_tokens;

        let completed = tokens.slice(-1)[0] == this.eot;
        tokens = new Tensor(
            [1, tokens.length],
            "int32",
            new Int32Array(tokens)
        );
        return [tokens, completed];
    };

    this.finalize = function (tokens, sum_logprobs) {
        // make sure each sequence has at least one EOT token at the end
        let newBuffer = new Int32Array(tokens.size() + 1);
        newBuffer.set(tokens.data);
        newBuffer[tokens.size()] = this.eot;
        tokens.shape[2] += 1;
        tokens.data = newBuffer;
        return [tokens, sum_logprobs];
    };
}

function SuppressBlank(tokenizer, sample_begin) {
    this.tokenizer = tokenizer;
    this.sample_begin = sample_begin;

    this.apply = function (logits, tokens) {
        if (tokens.shape[0] == this.sample_begin) {
            for (let i = 0; i < logits.length; i++) {
                // set negative infinity to blank token and EOT token
                logits[i][this.tokenizer.encode(" ") + [this.tokenizer.eot]] =
                    Number.NEGATIVE_INFINITY;
            }
        }
        return logits;
    };
}

function SuppressTokens(suppress_tokens) {
    this.suppress_tokens = suppress_tokens;
    this.apply = function (logits) {
        for (let j = 0; j < this.suppress_tokens.length; j++) {
            logits.data[this.suppress_tokens[j]] = Number.NEGATIVE_INFINITY;
        }
        return logits;
    };
}

function ApplyTimestampRules(
    tokenizer,
    sample_begin,
    max_initial_timestamp_index
) {
    this.tokenizer = tokenizer;
    this.sample_begin = sample_begin;
    this.max_initial_timestamp_index = max_initial_timestamp_index;

    this.apply = function (logits, tokens) {
        // suppress <|notimestamps|> which is handled by without_timestamps
        if (this.tokenizer.no_timestamps() != null) {
            logits.data[this.tokenizer.no_timestamps()] =
                Number.NEGATIVE_INFINITY;
        }

        // timestamps have to appear in pairs, except directly before EOT; mask logits accordingly
        let sampled_tokens = tokens.slice2d(this.sample_begin);
        let seq = [];
        for (let i = 0; i < sampled_tokens.length; i++) {
            seq.push(sampled_tokens[i]);
        }
        let last_was_timestamp =
            seq.length >= 1 &&
            seq.slice(-1)[0] >= this.tokenizer.timestamp_begin();
        let penultimate_was_timestamp =
            seq.length < 2 ||
            seq.slice(-2)[0] >= this.tokenizer.timestamp_begin();

        if (last_was_timestamp) {
            if (penultimate_was_timestamp) {
                // has to be non-timestamp
                for (let i = 0; i < logits.size(); i++) {
                    logits.data[i] = Number.NEGATIVE_INFINITY;
                }
            } else {
                // cannot be normal text tokens
                for (let i = 0; i < this.tokenizer.eot(); i++) {
                    logits.data[i] = Number.NEGATIVE_INFINITY;
                }
            }
        }

        let timestamps = [];
        for (let i = 0; i < sampled_tokens.size(); i++) {
            if (sampled_tokens.data[i] >= this.tokenizer.timestamp_begin()) {
                timestamps.push(sampled_tokens.data[i]);
            }
        }
        if (timestamps.length > 0) {
            // timestamps shouldn't decrease; forbid timestamp tokens smaller than the last
            for (
                let i = this.tokenizer.timestamp_begin();
                i < timestamps.slice(-1)[0];
                i++
            ) {
                logits.data[i] = Number.NEGATIVE_INFINITY;
            }
            // to force that timestamps are strictly increasing
            let timestamp_last;
            if (last_was_timestamp && !penultimate_was_timestamp) {
                timestamp_last = timestamps.slice(-1)[0];
            } else {
                timestamp_last = timestamps.slice(-1)[0] + 1;
            }
            for (
                let i = this.tokenizer.timestamp_begin();
                i < timestamp_last;
                i++
            ) {
                logits.data[i] = Number.NEGATIVE_INFINITY;
            }
        }

        if (tokens.size() == this.sample_begin) {
            // suppress generating non-timestamp tokens at the beginning
            for (let i = 0; i < this.tokenizer.timestamp_begin(); i++) {
                logits.data[i] = Number.NEGATIVE_INFINITY;
            }
            if (this.max_initial_timestamp_index != null) {
                // apply the `max_initial_timestamp` option
                let last_allowed =
                    this.tokenizer.timestamp_begin() +
                    this.max_initial_timestamp_index;
                for (let i = last_allowed + 1; i < logits.size(); i++) {
                    logits.data[i] = Number.NEGATIVE_INFINITY;
                }
            }
        }
        let logprob = [];
        for (let i = 0; i < logits.size(); i++) {
            logprob.push(logits.data[i]);
        }
        logprob = softmax(logprob);
        logprob = logprob.map((x) => Math.log(x));

        let logprobs = logprob;
        let timestamp_logprob = Math.log(
            logprobs
                .slice(this.tokenizer.timestamp_begin())
                .map((x) => Math.exp(x))
                .reduce((a, b) => a + b, 0)
        );
        let max_text_token_logprob = max(
            logprobs.slice(0, this.tokenizer.timestamp_begin())
        );
        if (timestamp_logprob > max_text_token_logprob) {
            for (let i = 0; i < this.tokenizer.timestamp_begin(); i++) {
                logits.data[i] = Number.NEGATIVE_INFINITY;
            }
        }
    };
}

function DecodingTask(model, options) {
    this.model = model;
    let language = options.language || "en";
    this.tokenizer = get_tokenizer(
        model.is_multilingual,
        language,
        options.task
    );
    this._verify_options = function (options) {
        if (options.beam_size !== null && options.best_of !== null) {
            throw new Error("beam_size and best_of can't be given together");
        }
        if (options.temperature == 0) {
            if (options.best_of !== null) {
                throw new Error(
                    "best_of with greedy sampling (T=0) is not compatible"
                );
            }
        }
        if (options.patience !== null && options.beam_size === null) {
            throw new Error("patience requires beam_size to be given");
        }
        if (
            options.length_penalty !== null &&
            !(0 <= options.length_penalty <= 1)
        ) {
            throw new Error(
                "length_penalty (alpha) should be a value between 0 and 1"
            );
        }
        return options;
    };
    this.options = this._verify_options(options);
    this.n_group = options.beam_size || options.best_of || 1;
    this.n_ctx = model.dims.n_text_ctx;
    this.sample_len = options.sample_len || model.dims.n_text_ctx / 2;
    this.sot_sequence = this.tokenizer.sot_sequence;
    if (this.options.without_timestamps) {
        this.sot_sequence = this.tokenizer.sot_sequence_including_notimestamps;
    }
    this._get_initial_tokens = function () {
        let tokens = this.sot_sequence.slice();
        let prefix = null;
        if ((prefix = this.options.prefix)) {
            let prefix_tokens = null;
            if (typeof prefix === "string") {
                prefix_tokens = this.tokenizer.encode(" " + prefix.trim());
            } else {
                prefix_tokens = prefix;
            }
            if (this.sample_len !== null) {
                let max_prefix_len = this.n_ctx / 2 - this.sample_len;
                prefix_tokens = prefix_tokens.slice(-max_prefix_len);
            }
            tokens = tokens.concat(prefix_tokens);
        }
        let prompt = null;
        if ((prompt = this.options.prompt)) {
            let prompt_tokens = null;
            if (typeof prompt === "string") {
                prompt_tokens = this.tokenizer.encode(" " + prompt.trim());
            } else {
                prompt_tokens = prompt;
            }
            tokens = [this.tokenizer.sot_prev()].concat(
                prompt_tokens.slice(-(this.n_ctx / 2 - 1)),
                tokens
            );
        }
        return tokens;
    };
    this.initial_tokens = this._get_initial_tokens();
    this.sample_begin = this.initial_tokens.length;
    this.sot_index = this.initial_tokens.indexOf(this.tokenizer.sot());
    this.inference = new Inference(model, this.initial_tokens.length);
    this.sequence_ranker = new MaximumLikelihoodRanker(options.length_penalty);
    this.decoder = new GreedyDecoder(options.temperature, this.tokenizer.eot());
    this.logit_filters = [];
    if (this.options.suppress_blank) {
        this.logit_filters.push(
            new SuppressBlank(model.tokenizer, this.sample_begin)
        );
    }
    this._get_suppress_tokens = function () {
        let suppress_tokens = this.options.suppress_tokens;
        if (typeof suppress_tokens === "string") {
            suppress_tokens = suppress_tokens
                .split(",")
                .map((t) => parseInt(t));
        }
        if (suppress_tokens.includes(-1)) {
            suppress_tokens = suppress_tokens.filter((t) => t >= 0);
            suppress_tokens = suppress_tokens.concat(
                this.tokenizer.non_speech_tokens()
            );
        } else if (suppress_tokens === null || suppress_tokens.length === 0) {
            suppress_tokens = []; // interpret empty string as an empty list
        } else {
            console.assert(
                Array.isArray(suppress_tokens),
                "suppress_tokens must be a list"
            );
        }
        suppress_tokens = suppress_tokens.concat([
            this.tokenizer.transcribe(),
            this.tokenizer.translate(),
            this.tokenizer.sot(),
            this.tokenizer.sot_prev(),
            this.tokenizer.sot_lm(),
        ]);
        if (this.tokenizer.no_speech() !== null) {
            // no-speech probability is collected separately
            suppress_tokens.push(this.tokenizer.no_speech());
        }
        return Array.from(new Set(suppress_tokens)).sort();
    };
    if (this.options.suppress_tokens) {
        this.logit_filters.push(
            new SuppressTokens(this._get_suppress_tokens())
        );
    }
    if (!options.without_timestamps) {
        let precision = this.model.chunk_length / model.dims.n_audio_ctx; // usually 0.02 seconds
        let max_initial_timestamp_index = null;
        if (options.max_initial_timestamp) {
            max_initial_timestamp_index = Math.round(
                this.options.max_initial_timestamp / precision
            );
        }
        this.logit_filters.push(
            new ApplyTimestampRules(
                this.tokenizer,
                this.sample_begin,
                max_initial_timestamp_index
            )
        );
    }
    this._get_initial_tokens = function () {
        let tokens = this.sot_sequence.slice();
        let prefix = null;
        if ((prefix = this.options.prefix)) {
            let prefix_tokens = null;
            if (typeof prefix === "string") {
                prefix_tokens = this.tokenizer.encode(" " + prefix.trim());
            } else {
                prefix_tokens = prefix;
            }
            if (this.sample_len !== null) {
                let max_prefix_len = this.n_ctx / 2 - this.sample_len;
                prefix_tokens = prefix_tokens.slice(-max_prefix_len);
            }
            tokens = tokens.concat(prefix_tokens);
        }
        let prompt = null;
        if ((prompt = this.options.prompt)) {
            let prompt_tokens = null;
            if (typeof prompt === "string") {
                prompt_tokens = this.tokenizer.encode(" " + prompt.trim());
            } else {
                prompt_tokens = prompt;
            }
            tokens = [this.tokenizer.sot_prev()].concat(
                prompt_tokens.slice(-(this.n_ctx / 2 - 1)),
                tokens
            );
        }
        return tokens;
    };

    this._get_audio_features = async function (mel) {
        let audio_features = null;
        if (
            mel.shape.slice(-2) ==
            [this.model.dims.n_audio_ctx, this.model.dims.n_audio_state]
        ) {
            // encoded audio features are given; skip audio encoding
            audio_features = mel;
            audio_features = new Tensor(
                audio_features.dims,
                audio_features.dtype,
                audio_features.data
            );
        } else {
            audio_features = await this.model.encoder(mel);
            audio_features = new Tensor(
                audio_features.shape,
                "float32",
                audio_features.data
            );
        }
        return audio_features;
    };

    this._detect_language = async function (audio_features, tokens) {
        let languages = new Array(audio_features.shape[0]).fill(
            this.options.language
        );
        let lang_tokens = null;
        let lang_probs = null;
        if (this.options.language === null || this.options.task === "lang_id") {
            [lang_tokens, lang_probs] = await this.model.detect_language(
                audio_features,
                this.tokenizer
            );
            languages = lang_probs.map((probs) => {
                return Object.keys(probs).reduce((a, b) =>
                    probs[a] > probs[b] ? a : b
                );
            });
            if (this.model.debug) {
                console.log(
                    "[DEBUG] Detected language:",
                    languages,
                    lang_probs
                );
            }
            if (this.options.language == null) {
                tokens.data[this.sot_index + 1] = lang_tokens;
            }
        }
        return [languages, lang_probs];
    };

    this._main_loop = async function (
        audio_features,
        tokens,
        tokenCallback,
        language
    ) {
        if (this.model.debug) {
            console.log("[DEBUG] Main decoding loop started.");
        }
        let n_batch = tokens.shape[0];
        let sum_logprobs = new Float32Array(n_batch);
        let no_speech_probs = new Float32Array(n_batch).fill(NaN);
        let completed = false;
        for (let i = 0; i < this.sample_len; i++) {
            let logits;
            if (i == 0) {
                for (let j = 0; j < tokens.shape[1]; j++) {
                    if (j == 0) {
                        let tmp = await this.inference.logits(
                            tokens.slice2d(j, j + 1),
                            audio_features
                        );
                        logits = new Tensor(
                            [tmp.shape[0], tokens.shape[1], tmp.shape[2]],
                            tmp.dtype
                        );
                        logits.data.set(tmp.data, 0);
                    } else {
                        let tmp = await this.inference.logits(
                            tokens.slice2d(j, j + 1),
                            audio_features
                        );
                        logits.data.set(tmp.data, tmp.size() * j);
                    }
                }
            } else {
                logits = await this.inference.logits(tokens, audio_features);
            }
            let no_speech = false;
            if (i == 0 && this.tokenizer.no_speech() !== null) {
                let probs_at_sot = logits
                    .reshape([logits.shape[1], logits.shape[2]])
                    .slice(this.sot_index, this.sot_index + 1)
                    .softmax2d();
                no_speech_probs = probs_at_sot.slice2d(
                    this.tokenizer.no_speech(),
                    this.tokenizer.no_speech() + 1
                );
                if (no_speech_probs.data[0] > 0.3) {
                    no_speech = true;
                }
            }
            logits = logits
                .reshape([logits.shape[1], logits.shape[2]])
                .slice(logits.shape[1] - 1, logits.shape[1]);
            for (let logit_filter of this.logit_filters) {
                logit_filter.apply(logits, tokens);
            }
            [tokens, completed] = await this.decoder.update(
                tokens,
                logits,
                sum_logprobs
            );
            // if tokens have enough length and the last token is timestamp, force eot
            if (
                no_speech ||
                (tokens.size() >= this.model.chunk_length * 4 &&
                    tokens.data[tokens.data.length - 1] >=
                        this.tokenizer.timestamp_begin()) ||
                tokens.size() >= this.model.chunk_length * 6
            ) {
                // push eot to tokens
                let newTokens = new Int32Array(tokens.data.length + 1);
                newTokens.set(tokens.data);
                newTokens[tokens.data.length] = this.tokenizer.eot();
                tokens.shape[1] += 1;
                tokens = new Tensor(tokens.shape, tokens.dtype, newTokens);
                completed = true;
            } else {
                // push eot to tokens
                let newTokens = new Int32Array(tokens.data.length + 1);
                newTokens.set(tokens.data);
                newTokens[tokens.data.length] = this.tokenizer.eot();
                newTokens = new Tensor(
                    [1, tokens.shape[1] + 1],
                    tokens.dtype,
                    newTokens
                );
                let newSumLogprobs = new Tensor(
                    [n_batch],
                    "float32",
                    sum_logprobs
                );
                newSumLogprobs = newSumLogprobs.reshape([1, this.n_group]);
                let _tokens;
                [_tokens, newSumLogprobs] = await this.decoder.finalize(
                    newTokens,
                    newSumLogprobs
                );
                newTokens = [];
                for (let i = 0; i < _tokens.shape[0]; i++) {
                    let s = _tokens.data;
                    let t = [];
                    let end = s
                        .map((x) => x == this.tokenizer.eot())
                        .indexOf(1);
                    for (let j = this.sample_begin; j < end; j++) {
                        t.push(s[j]);
                    }
                    newTokens.push(t);
                }
                newTokens = newTokens[0];
                let texts = [];
                texts.push(this.tokenizer.decode(newTokens));
                if (language != "en" && language != "ja") {
                    tokenCallback("");
                } else {
                    tokenCallback(texts[0]);
                }
            }
            if (completed || tokens.shape[1] > this.n_ctx) {
                break;
            }
        }
        return [
            tokens,
            new Tensor([n_batch], "float32", sum_logprobs),
            new Tensor([n_batch], "float32", no_speech_probs),
        ];
    };

    this.run = async function (mel, tokenCallback) {
        let n_audio = mel.shape[0];
        let audio_features = await this._get_audio_features(mel);
        let tokens = new Tensor(
            [1, this.initial_tokens.length],
            "int32",
            new Int32Array(this.initial_tokens)
        );
        let [languages, language_probs] = await this._detect_language(
            audio_features,
            tokens
        );
        if (this.options.task == "lang_id") {
            let results = [];
            for (let i = 0; i < audio_features.length; i++) {
                results.push(
                    new DecodingResult(
                        (audio_features = audio_features[i]),
                        (language = languages[i]),
                        (language_probs = language_probs[i])
                    )
                );
            }
            return results;
        }
        let sum_logprobs, no_speech_probs;
        [tokens, sum_logprobs, no_speech_probs] = await this._main_loop(
            audio_features,
            tokens,
            tokenCallback,
            languages[0]
        );
        console.assert(
            (audio_features.shape[0] == no_speech_probs.shape[0]) == n_audio
        );
        tokens = tokens.reshape([n_audio, 1, tokens.shape[1]]);
        sum_logprobs = sum_logprobs.reshape([n_audio, this.n_group]);
        let _tokens;
        [_tokens, sum_logprobs] = await this.decoder.finalize(
            tokens,
            sum_logprobs
        );
        tokens = [];
        for (let i = 0; i < _tokens.shape[0]; i++) {
            let s = _tokens.data;
            let t = [];
            let end = s.map((x) => x == this.tokenizer.eot()).indexOf(1);
            for (let j = this.sample_begin; j < end; j++) {
                t.push(s[j]);
            }
            tokens.push(t);
        }
        tokens = tokens[0];
        let texts = [];
        texts.push(this.tokenizer.decode(tokens));
        if (this.model.debug) {
            console.log("[DEBUG] Tokens:", tokens);
            console.log("[DEBUG] Texts:", texts);
        }
        let avg_logprobs = [];
        for (let i = 0; i < tokens.length; i++) {
            avg_logprobs.push(sum_logprobs[i] / (tokens[i].length + 1));
        }

        let results = [];
        for (let i = 0; i < texts.length; ++i) {
            results.push(
                new DecodingResult(
                    audio_features[i],
                    languages[i],
                    language_probs[i],
                    tokens[i],
                    texts[i],
                    avg_logprobs[i],
                    no_speech_probs,
                    this.options.temperature,
                    compressionRatio(texts[i])
                )
            );
        }
        return results;
    };
}

export function Whisper(
    modelType,
    preprocessorModel,
    encoderModel,
    decoderModel,
    positionalEmbedding,
    chunk_length = 30,
    debug = false
) {
    this.debug = debug;
    this.running = false;
    //console.log("pre threads", ort.env.wasm.numThreads);
    ort.env.wasm.numThreads = 4;
    this.encoderSession = ort.InferenceSession.create(encoderModel, {
        executionProviders: ["cpu"],
    });
    this.decoderSession = ort.InferenceSession.create(decoderModel, {
        executionProviders: ["cpu"],
    });
    this.preprocessorSession = ort.InferenceSession.create(preprocessorModel, {
        executionProviders: ["cpu"],
    });
    this.chunk_length = chunk_length;
    this.sample_rate = 16000;
    this.dims = {};
    this.dims.n_audio_state = modelType == "base" ? 512 : 768;
    this.dims.n_text_state = modelType == "base" ? 512 : 768;
    this.dims.n_audio_head = modelType == "base" ? 6 : 12;
    this.dims.n_text_head = modelType == "base" ? 6 : 12;
    this.dims.n_audio_layer = modelType == "base" ? 6 : 12;
    this.dims.n_text_layer = modelType == "base" ? 6 : 12;
    this.modelType = modelType;
    this.positional_embedding = (() => {
        let buffer = new Float32Array(positionalEmbedding);
        let array = [];
        for (let i = 0; i < 448; i++) {
            let row = [];
            for (let j = 0; j < this.dims.n_audio_state; j++) {
                row.push(buffer[i * this.dims.n_audio_state + j]);
            }
            array.push(row);
        }
        return array;
    })();
    this.dims.n_mels = 80;
    this.dims.n_audio_ctx = chunk_length * 50;
    this.dims.n_vocab = 51865;
    this.dims.n_text_ctx = 448;

    this.initializeCache = function () {
        this.self_attn_key_cache = new ort.Tensor(
            "float32",
            Float32Array.from(
                new Array(
                    this.dims.n_audio_layer * this.dims.n_audio_state
                ).fill(0)
            ),
            [this.dims.n_audio_layer, 1, this.dims.n_audio_state]
        );
        this.self_attn_value_cache = new ort.Tensor(
            "float32",
            Float32Array.from(
                new Array(
                    this.dims.n_audio_layer * this.dims.n_audio_state
                ).fill(0)
            ),
            [this.dims.n_audio_layer, 1, this.dims.n_audio_state]
        );
        this.cross_attn_key_cache = new ort.Tensor(
            "float32",
            Float32Array.from(
                new Array(
                    this.dims.n_audio_layer * this.dims.n_audio_state
                ).fill(0)
            ),
            [this.dims.n_audio_layer, 1, this.dims.n_audio_state]
        );
        this.cross_attn_value_cache = new ort.Tensor(
            "float32",
            Float32Array.from(
                new Array(
                    this.dims.n_audio_layer * this.dims.n_audio_state
                ).fill(0)
            ),
            [this.dims.n_audio_layer, 1, this.dims.n_audio_state]
        );
    };
    this.initializeCache();

    this.encoder = async function (x) {
        let start, end;
        let input = { input: new ort.Tensor("float32", x.data, x.shape) };
        if (this.debug || true) {
            start = performance.now();
            console.log("[DEBUG] Encoding started with input:", input);
        }
        let output = await this.encoderSession.then((session) =>
            session.run(input)
        );
        this.cross_attn_key_cache = output.key_cache;
        this.cross_attn_value_cache = output.value_cache;
        if (this.debug || true) {
            end = performance.now();
            console.log(
                `[DEBUG] Encoding finished in ${end - start}[ms] with output:`,
                output
            );
        }
        return new Tensor(
            output.output.dims,
            output.output.dtype,
            output.output.data
        );
    };

    this.decoder = async function (
        tokens,
        self_attn_key_cache,
        self_attn_value_cache,
        cross_attn_key_cache,
        cross_attn_value_cache,
        positional_embedding
    ) {
        let input = {
            input_token: new ort.Tensor("int32", tokens.data, [
                1,
                tokens.shape[1],
            ]),
            self_key_cache: self_attn_key_cache,
            self_value_cache: self_attn_value_cache,
            cross_key_cache: cross_attn_key_cache,
            cross_value_cache: cross_attn_value_cache,
            positional_embedding: new ort.Tensor(
                "float32",
                new Float32Array(positional_embedding.flat()),
                [positional_embedding.length, positional_embedding[0].length]
            ),
        };
        let start, end;
        if (this.debug) {
            start = performance.now();
            console.log("[DEBUG] Decoding started with input:", input);
        }
        let output = await this.decoderSession.then((session) =>
            session.run(input)
        );
        if (this.debug) {
            end = performance.now();
            console.log(
                `[DEBUG] Decoding finished in ${end - start}[ms] with output:`,
                output
            );
        }
        return [
            new Tensor(output.output.dims, "float32", output.output.data),
            output.key_cache,
            output.value_cache,
        ];
    };

    this.preprocessor = async function (x) {
        let input = { input: new ort.Tensor(x.dtype, x.data, x.shape) };
        let start, end;
        if (this.debug) {
            start = performance.now();
            console.log("[DEBUG] Preprocess started with input:", input);
        }
        let output = await this.preprocessorSession.then((session) =>
            session.run(input)
        );
        if (this.debug) {
            end = performance.now();
            console.log(
                `[DEBUG] Preprocess finished in ${
                    end - start
                }[ms] with output:`,
                output
            );
        }
        return new Tensor(
            output.output.dims,
            output.output.dtype,
            output.output.data
        );
    };

    this.logits = async function (tokens) {
        let offset = this.model.self_attn_value_cache.shape[1] - 1;
        let positional_embedding = (
            await this.model.positional_embedding
        ).slice(offset, offset + tokens.shape[1]);
        let [l, k, v] = await this.decoder(
            tokens,
            this.self_attn_key_cache,
            this.self_attn_value_cache,
            this.cross_attn_key_cache,
            this.cross_attn_value_cache,
            positional_embedding
        );
        this.self_attn_key_cache = k;
        this.self_attn_value_cache = v;
        return l;
    };

    this.detection_logits = async function (tokens) {
        let offset = this.self_attn_value_cache.dims[1] - 1;
        let positional_embedding = (await this.positional_embedding).slice(
            offset,
            offset + tokens.shape[1]
        );
        let [l] = await this.decoder(
            tokens,
            this.self_attn_key_cache,
            this.self_attn_value_cache,
            this.cross_attn_key_cache,
            this.cross_attn_value_cache,
            positional_embedding
        );
        return l;
    };

    this.is_multilingual = function () {
        return this.dims.n_vocab == 51865;
    };

    this.detect_language = async function (mel, tokenizer) {
        if (tokenizer == null) {
            tokenizer = get_tokenizer(this.is_multilingual());
        }
        let single = mel.shape.length == 2;
        if (single) {
            mel = mel.reshape([1, mel.shape[0], mel.shape[1]]);
        }

        let n_audio = mel.shape[0];
        let x = new Tensor(
            [1, n_audio],
            "int32",
            new Int32Array(n_audio).fill(tokenizer.sot())
        );
        let logits = await this.detection_logits(x);
        logits = logits.reshape([1, logits.shape[2]]);

        let mask = new ort.Tensor("bool", new Array(logits.shape[1]).fill(1), [
            logits.shape[1],
        ]);
        let indices = tokenizer.all_language_tokens();
        let all_language_tokens = indices;
        let all_language_codes = tokenizer.all_language_codes();
        for (let index of indices) {
            mask.data[index] = false;
        }
        for (let i = 0; i < logits.shape[1]; i++) {
            if (mask.data[i]) {
                logits.data[i] = Number.NEGATIVE_INFINITY;
            }
        }
        let language_tokens = logits.data.indexOf(Math.max(...logits.data));
        let language_token_probs = softmax(logits.data);
        let language_probs = [];
        for (let i = 0; i < n_audio; i++) {
            let prob = {};
            for (let j = 0; j < indices.length; j++) {
                let token = all_language_tokens[j];
                let code = all_language_codes[j];
                prob[code] = language_token_probs[token];
            }
            language_probs.push(prob);
        }
        if (single) {
            language_tokens = language_tokens[0];
            language_probs = language_probs[0];
        }
        return [language_tokens, language_probs];
    };

    this.load_audio = async function (audioUrl) {
        var audio = new Audio();
        audio.src = audioUrl;
        const ctx = new AudioContext({ sampleRate: 16000 });
        var audioBuffer = await fetch(audioUrl)
            .then((response) => response.arrayBuffer())
            .then((buffer) => ctx.decodeAudioData(buffer))
            .then((decoded) => {
                let audio = decoded;
                let buffer = Float32Array.from(new Array(audio.length).fill(0));
                if (audio.numberOfChannels > 1) {
                    let channel0 = audio.getChannelData(0);
                    let channel1 = audio.getChannelData(1);
                    for (let j = 0; j < channel0.length; j++) {
                        buffer[j] = (channel0[j] + channel1[j]) / 2;
                    }
                } else {
                    let channel = audio.getChannelData(0);
                    for (let j = 0; j < channel.length; j++) {
                        buffer[j] = channel[j];
                    }
                }
                return buffer;
            });
        return new Tensor([audioBuffer.length], "float32", audioBuffer);
    };

    // tensor: [n_samples]
    this.pad_or_trim = function (tensor, length) {
        if (length == null || length === undefined) {
            length = this.chunk_length * this.sample_rate;
        }
        if (tensor.size() > length) {
            tensor = tensor.slice(tensor.size() - length, tensor.size());
        }
        if (tensor.size() < length) {
            tensor = tensor.pad(length - tensor.shape[0]);
        }
        tensor = tensor.pad(8000).slice(8000);
        return tensor;
    };

    this.decode = async function (mel, options, tokenCallback = () => {}) {
        if (mel.shape.length == 2) {
            mel = mel.reshape([1, mel.shape[0], mel.shape[1]]);
        }
        let task = new DecodingTask(this, options);
        let result = await task.run(mel, tokenCallback);
        return result[0];
    };
}
