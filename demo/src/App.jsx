import { useState } from "react";
import reactLogo from "./assets/react.svg";
import viteLogo from "/vite.svg";
import "./App.css";
import audioUrl from "./assets/en.mp3";
import { Whisper, DecodingOptions, Tensor } from "@core/whisper";
import positionalEmbeddingBaseUrl from "@models/positional_embedding_base.bin.gz";
import positionalEmbeddingSmallUrl from "@models/positional_embedding_small.bin.gz";
import processorCode from "./whisper_processor.js?raw";

const CHUNK_LENGTH = 10;
let model_type = "small";
import preprocessorBaseModelUrl from "@models/preprocessor_base_5s_int8.onnx.gz";
import preprocessorSmallModelUrl from "@models/preprocessor_small_10s_int8.onnx.gz";
import encoderBaseModelUrl from "@models/encoder_base_5s_int8.onnx.gz";
import encoderSmallModelUrl from "@models/encoder_small_10s_int8.onnx.gz";
import decoderBaseModelUrl from "@models/decoder_base_5s_int8.onnx.gz";
import decoderSmallModelUrl0 from "@models/decoder_small_10s_int8.onnx.0.gz";
import decoderSmallModelUrl1 from "@models/decoder_small_10s_int8.onnx.1.gz";

async function convert() {
    let whisper = new Whisper(
        model_type,
        preprocessorSmallModelUrl,
        encoderSmallModelUrl,
        [decoderSmallModelUrl0, decoderSmallModelUrl1],
        positionalEmbeddingSmallUrl,
        CHUNK_LENGTH,
        true
    );
    if (true) {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: true,
        });
        const tmpCtx = new AudioContext({ sampleRate: 16000 });
        const source = tmpCtx.createMediaStreamSource(stream);
        const processorBlob = new Blob([processorCode], {
            type: "text/javascript",
        });
        const processorUrl = URL.createObjectURL(processorBlob);
        await tmpCtx.audioWorklet.addModule(processorUrl).then(() => {
            const node = new AudioWorkletNode(tmpCtx, "whisper-processor", {
                processorOptions: {
                    chunkLength: CHUNK_LENGTH,
                },
            });
            node.port.onmessage = (e) => {
                let buffer = new Float32Array(e.data);
                let tensor = new Tensor([buffer.length], "float32", buffer);
                let audio = whisper.pad_or_trim(tensor);
                whisper
                    .preprocessor(audio)
                    .then((mel) => whisper.decode(mel, new DecodingOptions()))
                    .then((result) => {
                        console.log("result", result);
                    });
            };
            source.connect(node);
            //source.connect(whisperProcessor).connect(context.destination);
        });
    }
    //const [track] = stream.getAudioTracks();
    return;
    let audio = await whisper.load_audio(audioUrl);
    audio = whisper.pad_or_trim(audio);
    let mel = await whisper.preprocessor(audio);
    let options = new DecodingOptions();
    let result = await whisper.decode(mel, options);
    return result;
}

function App() {
    const [count, setCount] = useState(0);

    return (
        <div className="App">
            <div>
                <a href="https://vitejs.dev" target="_blank" rel="noreferrer">
                    <img src={viteLogo} className="logo" alt="Vite logo" />
                </a>
                <a href="https://reactjs.org" target="_blank" rel="noreferrer">
                    <img
                        src={reactLogo}
                        className="logo react"
                        alt="React logo"
                    />
                </a>
            </div>
            <h1>Vite + React</h1>
            <div className="card">
                <button
                    onClick={() => {
                        setCount((count) => count + 1);
                        convert();
                        //source.start(0);
                    }}
                >
                    count is {count}
                </button>
                <p>
                    Edit <code>src/App.jsx</code> and save to test HMR
                </p>
            </div>
            <p className="read-the-docs">
                Click on the Vite and React logos to learn more
            </p>
        </div>
    );
}

export default App;
