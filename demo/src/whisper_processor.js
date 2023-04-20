class WhisperProcessor extends AudioWorkletProcessor {
    // Custom AudioParams can be defined with this static getter.
    static get parameterDescriptors() {
        return [{ name: "hoge", defaultValue: 1 }];
    }

    constructor(options) {
        super();
        //console.log("options", options);
        this.buffer = [];
        this.chunkLength = options.processorOptions.chunkLength;
        this.first = true;
    }

    process(inputs, outputs, parameters) {
        let input = inputs[0];
        //let output = outputs[0];
        let gain = parameters.hoge;
        for (let i = 0; i < input[0].length; i++) {
            this.buffer.push(input[0][i]);
        }
        if (this.buffer.length >= this.chunkLength * 16000 && this.first) {
            this.port.postMessage(
                this.buffer.slice(0, this.chunkLength * 16000)
            );
            this.buffer = this.buffer.slice(this.chunkLength * 8000);
            //this.first = false;
        }
        return true;
    }
}

registerProcessor("whisper-processor", WhisperProcessor);
