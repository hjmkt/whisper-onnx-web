import { Whisper, DecodingOptions, Tensor } from "@core/whisper";

const whisperWorker = {
    whisper: null,
};

self.onmessage = function (e) {
    if (e.data.type === "init") {
        whisperWorker.whisper = new Whisper(e.data.modelType, e.data.preprocessorModel, e.data.encoderModel, e.data.decoderModel, e.data.positionalEmbeddingModel, e.data.chunkLength, e.data.debug);
    } else if (e.data.type === "process") {
        whisperWorker.whisper.process(e.data.buffer);
    }
};
