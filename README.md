# whisper-onnx-web

```javascript
let modelType = "small";

// Models as ArrayBuffer
let embeddingModel = await fetch(URL_OF_EMBEDDING_MODEL);
let preprocessorModel = await fetch(URL_OF_PREPROCESSOR_MODEL);
let encoderModel = await fetch(URL_OF_ENCODER_MODEL);
let decoderModel = await fetch(URL_OF_DECODER_MODEL);

let chunkLength = 10;
let debug = false;

let whisper = new Whisper(
    modelType,
    preprocessorModel,
    encoderModel,
    decoderModel,
    embeddingModel,
    chunkLength,
    debug
);

// Prepare audio data at 16,000Hz as an array of values within the range of [-1.0, 1.0].
let audio = whisper.load_audio(URL_OF_AUDI_FILE);

let tokenCallback = (tokens) => {
    console.log(tokens);
};

let text = await whisper
    .preprocessor(audio)
    .then((mel) => whisper.decode(mel, new DecodingOptions(), tokenCallback))
    .then((result) => result.text);

console.log(text);
// "Hello world."
```
