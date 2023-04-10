import { vocab } from "./vocab";
import { special_tokens } from "./special_tokens";
import { non_speech_tokens } from "./non_speech_tokens";

let id_to_token = new Array(Object.keys(vocab).length);
for (let [token, id] of Object.entries(vocab)) {
    id_to_token[id] = token;
}

let bs = [];
for (let b = "!".charCodeAt(0); b <= "~".charCodeAt(0); b++) {
    bs.push(b);
}
for (let b = 161; b <= 172; b++) {
    bs.push(b);
}
for (let b = 174; b <= 255; b++) {
    bs.push(b);
}
let cs = bs.slice(0, bs.length);
let n = 0;
for (let b = 0; b <= 255; b++) {
    if (!bs.includes(b)) {
        bs.push(b);
        cs.push(256 + n);
        n += 1;
    }
}
const char_to_byte = Object.fromEntries(bs.map((b, i) => [String.fromCharCode(cs[i]), b]));

const LANGUAGES = {
    en: "english",
    zh: "chinese",
    de: "german",
    es: "spanish",
    ru: "russian",
    ko: "korean",
    fr: "french",
    ja: "japanese",
    pt: "portuguese",
    tr: "turkish",
    pl: "polish",
    ca: "catalan",
    nl: "dutch",
    ar: "arabic",
    sv: "swedish",
    it: "italian",
    id: "indonesian",
    hi: "hindi",
    fi: "finnish",
    vi: "vietnamese",
    he: "hebrew",
    uk: "ukrainian",
    el: "greek",
    ms: "malay",
    cs: "czech",
    ro: "romanian",
    da: "danish",
    hu: "hungarian",
    ta: "tamil",
    no: "norwegian",
    th: "thai",
    ur: "urdu",
    hr: "croatian",
    bg: "bulgarian",
    lt: "lithuanian",
    la: "latin",
    mi: "maori",
    ml: "malayalam",
    cy: "welsh",
    sk: "slovak",
    te: "telugu",
    fa: "persian",
    lv: "latvian",
    bn: "bengali",
    sr: "serbian",
    az: "azerbaijani",
    sl: "slovenian",
    kn: "kannada",
    et: "estonian",
    mk: "macedonian",
    br: "breton",
    eu: "basque",
    is: "icelandic",
    hy: "armenian",
    ne: "nepali",
    mn: "mongolian",
    bs: "bosnian",
    kk: "kazakh",
    sq: "albanian",
    sw: "swahili",
    gl: "galician",
    mr: "marathi",
    pa: "punjabi",
    si: "sinhala",
    km: "khmer",
    sn: "shona",
    yo: "yoruba",
    so: "somali",
    af: "afrikaans",
    oc: "occitan",
    ka: "georgian",
    be: "belarusian",
    tg: "tajik",
    sd: "sindhi",
    gu: "gujarati",
    am: "amharic",
    yi: "yiddish",
    lo: "lao",
    uz: "uzbek",
    fo: "faroese",
    ht: "haitian creole",
    ps: "pashto",
    tk: "turkmen",
    nn: "nynorsk",
    mt: "maltese",
    sa: "sanskrit",
    lb: "luxembourgish",
    my: "myanmar",
    bo: "tibetan",
    tl: "tagalog",
    mg: "malagasy",
    as: "assamese",
    tt: "tatar",
    haw: "hawaiian",
    ln: "lingala",
    ha: "hausa",
    ba: "bashkir",
    jw: "javanese",
    su: "sundanese"
};

const TO_LANGUAGE_CODE1 = Object.fromEntries(Object.entries(LANGUAGES).map(([k, v]) => [v, k]));
const TO_LANGUAGE_CODE2 = {
    burmese: "my",
    valencian: "ca",
    flemish: "nl",
    haitian: "ht",
    letzeburgesch: "lb",
    pushto: "ps",
    panjabi: "pa",
    moldavian: "ro",
    moldovan: "ro",
    sinhalese: "si",
    castilian: "es"
};
const TO_LANGUAGE_CODE = { ...TO_LANGUAGE_CODE1, ...TO_LANGUAGE_CODE2 };

function Tokenizer(encoding, language = null, task = null) {
    this.encoding = encoding;
    this.language = language;
    this.task = task;
    this.special_tokens = special_tokens;
    let sot = this.special_tokens["<|startoftranscript|>"];
    let translate = this.special_tokens["<|translate|>"];
    let transcribe = this.special_tokens["<|transcribe|>"];
    let langs = Object.keys(LANGUAGES);
    let sot_sequence = [sot];
    if (this.language) {
        sot_sequence.push(sot + 1 + langs.indexOf(this.language));
    }
    if (this.task) {
        let task_token = this.task == "transcribe" ? transcribe : translate;
        sot_sequence.push(task_token);
    }
    this.sot_sequence = sot_sequence;

    this.encode = function (text) {
        return this.encoding.encode(text);
    };

    this.decode = function (token_ids) {
        token_ids = token_ids.filter((t) => t < this.timestamp_begin());
        return this.encoding.decode(token_ids).replace(/Ġ/g, " ").replace(/^\s+/g, "");
    };

    this.decode_with_timestamps = function (token_ids) {
        return this.encoding.decode(token_ids);
    };

    this.eot = function () {
        return 50257;
    };
    this.transcribe = function () {
        return this.special_tokens["<|transcribe|>"];
    };
    this.translate = function () {
        return this.special_tokens["<|translate|>"];
    };
    this.sot = function () {
        return this.special_tokens["<|startoftranscript|>"];
    };
    this.sot_lm = function () {
        return this.special_tokens["<|startoflm|>"];
    };
    this.sot_prev = function () {
        return this.special_tokens["<|startofprev|>"];
    };
    this.no_speech = function () {
        return this.special_tokens["<|nospeech|>"];
    };
    this.no_timestamps = function () {
        return this.special_tokens["<|notimestamps|>"];
    };
    this.timestamp_begin = function () {
        return this.special_tokens["<|0.00|>"];
    };
    this.language_token = function () {
        if (this.language === null) {
            throw new Error("This tokenizer does not have language token configured");
        }
        let token;
        if ((token = this.special_tokens[`<|${this.language}|>`])) {
            return token;
        }
        throw new Error(`Language ${this.language} not found in tokenizer.`);
    };
    this.all_language_tokens = function () {
        let result = [];
        for (let [token, token_id] of Object.entries(this.special_tokens)) {
            if (/\d/.test(token)) {
                continue;
            }
            if (token.replace(/[<|>]/g, "") in LANGUAGES) {
                result.push(token_id);
            }
        }
        return result;
    };
    this.all_language_codes = function () {
        let result = [];
        for (let token of Object.keys(this.special_tokens)) {
            if (/\d/.test(token)) {
                continue;
            }
            if (token.replace(/[<|>]/g, "") in LANGUAGES) {
                result.push(token.replace(/[<|>]/g, ""));
            }
        }
        return result;
    };
    this.sot_sequence_including_notimestamps = function () {
        return this.sot_sequence.concat([this.no_timestamps()]);
    };
    this.non_speech_tokens = function () {
        return non_speech_tokens;
    };
    this.split_to_word_tokens = function (tokens) {
        if (this.language in ["zh", "ja", "th", "lo", "my"]) {
            return this.split_tokens_on_unicode(tokens);
        }
        return this.split_tokens_on_spaces(tokens);
    };
    this.split_tokens_on_unicode = function (tokens) {
        let decoded_full = this.decode_with_timestamps(tokens);
        let replacement_char = "\ufffd";

        let words = [];
        let word_tokens = [];
        let current_tokens = [];
        let unicode_offset = 0;

        for (let token of tokens) {
            current_tokens.push(token);
            let decoded = this.decode_with_timestamps(current_tokens);

            if (
                !decoded.includes(replacement_char) ||
                decoded_full[unicode_offset + decoded.index(replacement_char)] == replacement_char
            ) {
                words.push(decoded);
                word_tokens.push(current_tokens);
                current_tokens = [];
                unicode_offset += decoded.length;
            }
        }
        return [words, word_tokens];
    };
    this.split_tokens_on_spaces = function (tokens) {
        let [subwords, subword_tokens] = this.split_tokens_on_unicode(tokens);
        let words = [];
        let word_tokens = [];
        for (let i = 0; i < subwords.length; i++) {
            let subword = subwords[i];
            let subword_token = subword_tokens[i];
            let special = subword_token[0] >= this.eot;
            let with_space = subword.startsWith(" ");
            let punctuation = subword.trim() in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
            if (special || with_space || punctuation || words.length == 0) {
                words.push(subword);
                word_tokens.push(subword_token);
            } else {
                words[words.length - 1] = words[words.length - 1] + subword;
                word_tokens[word_tokens.length - 1].extend(subword_token);
            }
        }
        return [words, word_tokens];
    };
}

export function get_tokenizer(multilingual, language, task) {
    if (language) {
        language = language.toLowerCase();
        if (!LANGUAGES[language]) {
            if (TO_LANGUAGE_CODE[language]) {
                language = TO_LANGUAGE_CODE[language];
            } else {
                throw new Error(`Unsupported language: ${language}`);
            }
        }
    }
    if (multilingual) {
        language = language || "en";
        task = task || "transcribe";
    } else {
        language = null;
        task = null;
    }
    let encoding = get_encoding();
    return new Tokenizer(encoding, language, task);
}

function Coder() {
    this.decode = function (ids) {
        let tokens = [];
        for (let id of ids) {
            let token = id_to_token[id];
            tokens.push(token);
        }
        let decoded = tokens.join("");
        let utf8 = new Uint8Array(new ArrayBuffer(decoded.length));
        for (let i = 0; i < decoded.length; i++) {
            utf8[i] = char_to_byte[decoded.charAt(i)];
        }
        let str = new TextDecoder("utf-8").decode(utf8);
        return str;
    };
}

function get_encoding() {
    return new Coder();
}
