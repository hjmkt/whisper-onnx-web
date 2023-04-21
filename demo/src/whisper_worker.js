
const whisperWorker = {
    whisper: null,
};

const originalFetch = self.fetch;
const basePath = import.meta.env.BASE_URL;
self.fetch = function (input, init) {
    console.log("start", input, basePath);
    if (typeof input === "string" && input.match(/ort.*.wasm/g)) {
        input = `${basePath}${input}`.replace(/\/\//g, "/");
        console.log("fetch", input);
    }

    return originalFetch(input, init);
};
import { Whisper, DecodingOptions, Tensor } from "@core/whisper";

self.onmessage = function (e) {
    console.log("worker received message", e);
    if (e.data.type === "init") {
        whisperWorker.whisper = new Whisper(
            e.data.modelType,
            e.data.preprocessorModel,
            e.data.encoderModel,
            e.data.decoderModel,
            e.data.positionalEmbedding,
            e.data.chunkLength,
            e.data.debug
        );
    } else if (e.data.type === "run") {
        let tensor = new Tensor(
            e.data.tensor.shape,
            e.data.tensor.dtype,
            e.data.tensor.data
        );
        console.log("run tensor", tensor);
        let audio = whisperWorker.whisper.pad_or_trim(tensor);
        whisperWorker.whisper
            .preprocessor(audio)
            .then((mel) =>
                whisperWorker.whisper.decode(
                    mel,
                    new DecodingOptions(),
                    (tokens) =>
                        self.postMessage({ action: "tokens", payload: tokens })
                )
            )
            .then((result) => {
                console.log("result", result);
                if (result.language != "en" && result.language != "ja") {
                    result.text = "";
                }
                self.postMessage({
                    action: "result",
                    modelType: whisperWorker.whisper.modelType,
                    payload: result.text,
                });
                whisperWorker.whisper.initializeCache();
            });
    }
};
