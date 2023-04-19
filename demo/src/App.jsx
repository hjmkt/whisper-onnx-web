import { useState, useEffect } from "react";
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

let whisper_pool = {
    base: [
        new Whisper(
            "base",
            preprocessorBaseModelUrl,
            encoderBaseModelUrl,
            [decoderBaseModelUrl],
            positionalEmbeddingBaseUrl,
            5,
            true
        ),
        new Whisper(
            "base",
            preprocessorBaseModelUrl,
            encoderBaseModelUrl,
            [decoderBaseModelUrl],
            positionalEmbeddingBaseUrl,
            5,
            true
        ),
    ],
    small: [
        new Whisper(
            "small",
            preprocessorSmallModelUrl,
            encoderSmallModelUrl,
            [decoderSmallModelUrl0, decoderSmallModelUrl1],
            positionalEmbeddingSmallUrl,
            10,
            true
        ),
        new Whisper(
            "small",
            preprocessorSmallModelUrl,
            encoderSmallModelUrl,
            [decoderSmallModelUrl0, decoderSmallModelUrl1],
            positionalEmbeddingSmallUrl,
            10,
            true
        ),
    ],
};

async function getWhisper(model_type) {
    let whisper = null;
    while (whisper == null) {
        for (let m of whisper_pool[model_type]) {
            if (!m.running) {
                whisper = m;
                break;
            }
        }
        if (whisper == null) {
            await new Promise((resolve) => setTimeout(resolve, 100));
        }
    }
    return whisper;
}

async function runWhisper(
    model_type,
    tensor,
    textCallback,
    tokenCallback = () => {}
) {
    getWhisper(model_type).then((whisper) => {
        let audio = whisper.pad_or_trim(tensor);
        console.log("cb", tokenCallback);
        return whisper
            .preprocessor(audio)
            .then((mel) =>
                whisper.decode(mel, new DecodingOptions(), tokenCallback)
            )
            .then((result) => {
                console.log("result", result);
                textCallback(result.text);
            });
    });
}

async function convert(setText) {
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
                console.log("base start");
                runWhisper("base", tensor, setText, setText);
                console.log("base end");
                //tensor = new Tensor([buffer.length], "float32", buffer);
                //runWhisper("small", tensor, setText);
            };
            source.connect(node);
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
    const [text, setText] = useState("");
    let texts = [""];

    const updateText = (text) => {
        texts[0] = text;
        //console.log("texts", texts[0]);
    };

    //useEffect(() => {
    //const interval = setInterval(() => {
    ////texts[0] = texts[0] + "a";
    //setText(texts[0]);
    //console.log("texts", texts[0]);
    //}, 100);
    //return () => clearInterval(interval);
    //}, []);

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
            <textarea
                readOnly
                className="textarea"
                rows="6"
                cols="150"
                onChange={(e) => setText(e.target.value)}
                value={text}
            />
            <div className="card">
                <button
                    onClick={() => {
                        setCount((count) => count + 1);
                        convert(updateText);
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
