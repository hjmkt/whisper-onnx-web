//const CHUNK_LENGTH = 10;

//async function convert() {
//let whisper = new Whisper(
//preprocessorWasmModelUrl,
//encoderWasmModelUrl,
//decoderWasmModelUrl,
//positionalEmbeddingUrl,
//CHUNK_LENGTH,
//true
//);
//const stream = await navigator.mediaDevices.getUserMedia({
//video: false,
//audio: true,
//});
//const [track] = stream.getAudioTracks();
//const tmpCtx = new AudioContext({ sampleRate: 16000 });
//let audio = await whisper.load_audio(audioUrl);
//audio = whisper.pad_or_trim(audio);
//let mel = await whisper.preprocessor(audio);
//let options = new DecodingOptions();
//let result = await whisper.decode(mel, options);
//return result;
//}

class WhisperProcessor extends AudioWorkletProcessor {
    // Custom AudioParams can be defined with this static getter.
    static get parameterDescriptors() {
        return [{ name: "hoge", defaultValue: 1 }];
    }

    constructor(options) {
        super();
        console.log("options", options);
        this.buffer = [];
        this.chunkLength = options.processorOptions.chunkLength;
        this.first = true;
    }

    process(inputs, outputs, parameters) {
        let input = inputs[0];
        //let output = outputs[0];
        let gain = parameters.hoge;
        //this.port.postMessage("Hello from worker");
        for (let i = 0; i < input[0].length; i++) {
            this.buffer.push(input[0][i]);
        }
        if (
            this.buffer.length >= (this.chunkLength - 0) * 16000 &&
            this.first
        ) {
            this.port.postMessage(
                this.buffer.slice(0, (this.chunkLength - 0) * 16000)
            );
            this.buffer = this.buffer.slice((this.chunkLength - 0) * 16000);
            this.first = false;
        }
        //for (let channel = 0; channel < input.length; ++channel) {
        //let inputChannel = input[channel];
        //let outputChannel = output[channel];
        //for (let i = 0; i < inputChannel.length; ++i)
        //outputChannel[i] = inputChannel[i] * gain[i];
        //}

        return true;
    }
}

registerProcessor("whisper-processor", WhisperProcessor);
