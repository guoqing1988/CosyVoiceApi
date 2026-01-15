const { createApp, ref, onMounted, computed } = Vue;

createApp({
    setup() {
        const form = ref({
            text: "我是通义实验室语音团队全新推出的生成式语音大模型,提供舒适自然的语音合成能力。",
            mode: "sft",
            speaker: "",
            prompt_text: "",
            prompt_wav_path: "",
            instruct_text: "",
            source_wav_path: "",
            stream: false,
            speed: 1.0
        });

        const health = ref({});
        const speakers = ref([]);
        const voices = ref([]);
        const loading = ref(false);
        const audioUrl = ref(null);
        const audioSampleRate = ref(22050);
        const error = ref(null);
        const audioPlayer = ref(null);
        let abortController = null;

        const statusClass = computed(() => {
            if (health.value.status === 'ok') return 'bg-green-500';
            if (health.value.status === 'error') return 'bg-red-500';
            return 'bg-yellow-500';
        });

        const fetchHealth = async () => {
            try {
                const res = await fetch('/v1/health');
                health.value = await res.json();
            } catch (e) {
                health.value = { status: 'error' };
            }
        };

        const fetchSpeakers = async () => {
            try {
                const res = await fetch('/v1/speakers');
                const data = await res.json();
                speakers.value = data.speakers;
                if (speakers.value.length > 0 && !form.value.speaker) {
                    form.value.speaker = speakers.value[0];
                }
            } catch (e) {
                console.error("Failed to fetch speakers", e);
            }
        };

        const fetchVoices = async () => {
            try {
                const res = await fetch('/v1/voices');
                const data = await res.json();
                console.log(data);
                voices.value = data.voices;
                // if (voices.value.length > 0 && !form.value.voice_id) {
                //     form.value.voice_id = voices.value[0];
                // }
            } catch (e) {
                console.error("Failed to fetch voices", e);
            }
        };

        const generateAudio = async () => {
            loading.value = true;
            error.value = null;
            audioUrl.value = null;
            abortController = new AbortController();

            try {
                if (form.value.stream) {
                    await handleStreaming();
                } else {
                    await handleSync();
                }
            } catch (e) {
                if (e.name !== 'AbortError') {
                    error.value = e.message;
                }
            } finally {
                loading.value = false;
                abortController = null;
            }
        };

        const handleSync = async () => {
            const res = await fetch('/v1/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(form.value),
                signal: abortController?.signal
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '请求失败');
            }

            const data = await res.json();
            audioSampleRate.value = data.sample_rate;
            audioUrl.value = `data:audio/wav;base64,${data.audio}`;
        };

        const handleStreaming = async () => {
            // Use Web Audio API for streaming playback
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const sampleRate = 22050; // CosyVoice default sample rate

            const res = await fetch('/v1/tts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...form.value, stream: true }),
                signal: abortController?.signal
            });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || '流式请求失败');
            }

            const reader = res.body.getReader();
            const audioBuffers = [];
            let startTime = audioContext.currentTime;
            let hasStartedPlaying = false;
            let pendingBytes = new Uint8Array(0); // Buffer for incomplete chunks

            const playAudioChunk = (float32Array) => {
                if (float32Array.length === 0) return;

                const audioBuffer = audioContext.createBuffer(1, float32Array.length, sampleRate);
                audioBuffer.getChannelData(0).set(float32Array);

                const source = audioContext.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(audioContext.destination);

                if (!hasStartedPlaying) {
                    startTime = audioContext.currentTime;
                    hasStartedPlaying = true;
                }

                source.start(startTime);
                startTime += audioBuffer.duration;
            };

            try {
                while (true) {
                    const { done, value } = await reader.read();

                    if (done) {
                        // Process any remaining bytes
                        if (pendingBytes.length >= 4) {
                            const completeLength = Math.floor(pendingBytes.length / 4) * 4;
                            const float32Array = new Float32Array(pendingBytes.buffer.slice(0, completeLength));
                            playAudioChunk(float32Array);
                            audioBuffers.push(float32Array);
                        }
                        break;
                    }

                    // Combine pending bytes with new data
                    const combined = new Uint8Array(pendingBytes.length + value.length);
                    combined.set(pendingBytes, 0);
                    combined.set(value, pendingBytes.length);

                    // Calculate how many complete Float32 values we have (4 bytes each)
                    const completeLength = Math.floor(combined.length / 4) * 4;

                    if (completeLength > 0) {
                        // Convert complete bytes to Float32Array
                        const float32Array = new Float32Array(combined.buffer.slice(0, completeLength));

                        // Play immediately when we receive data
                        playAudioChunk(float32Array);
                        audioBuffers.push(float32Array);
                    }

                    // Store remaining incomplete bytes for next iteration
                    pendingBytes = combined.slice(completeLength);
                }

                // After streaming is complete, create a downloadable blob
                const totalLength = audioBuffers.reduce((sum, arr) => sum + arr.length, 0);
                const combinedArray = new Float32Array(totalLength);
                let offset = 0;
                for (const arr of audioBuffers) {
                    combinedArray.set(arr, offset);
                    offset += arr.length;
                }

                // Create WAV file for download
                const wavBlob = createWavBlob(combinedArray, sampleRate);
                audioUrl.value = URL.createObjectURL(wavBlob);
                audioSampleRate.value = sampleRate;
            } catch (e) {
                console.error('Streaming error:', e);
                throw new Error('流式播放失败: ' + e.message);
            }
        };

        // Helper function to create WAV blob from Float32Array
        const createWavBlob = (samples, sampleRate) => {
            const buffer = new ArrayBuffer(44 + samples.length * 2);
            const view = new DataView(buffer);

            // WAV header
            const writeString = (offset, string) => {
                for (let i = 0; i < string.length; i++) {
                    view.setUint8(offset + i, string.charCodeAt(i));
                }
            };

            writeString(0, 'RIFF');
            view.setUint32(4, 36 + samples.length * 2, true);
            writeString(8, 'WAVE');
            writeString(12, 'fmt ');
            view.setUint32(16, 16, true);
            view.setUint16(20, 1, true); // PCM
            view.setUint16(22, 1, true); // Mono
            view.setUint32(24, sampleRate, true);
            view.setUint32(28, sampleRate * 2, true);
            view.setUint16(32, 2, true);
            view.setUint16(34, 16, true);
            writeString(36, 'data');
            view.setUint32(40, samples.length * 2, true);

            // Convert float32 to int16
            let offset = 44;
            for (let i = 0; i < samples.length; i++) {
                const s = Math.max(-1, Math.min(1, samples[i]));
                view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                offset += 2;
            }

            return new Blob([buffer], { type: 'audio/wav' });
        };

        const stopGeneration = () => {
            if (abortController) {
                abortController.abort();
            }
            loading.value = false;
        };

        onMounted(() => {
            fetchHealth();
            // fetchSpeakers();
            fetchVoices();
            // Refresh health every 30s
            setInterval(fetchHealth, 30000);
        });

        return {
            form,
            health,
            speakers,
            voices,
            loading,
            audioUrl,
            audioSampleRate,
            error,
            audioPlayer,
            statusClass,
            generateAudio,
            stopGeneration
        };
    }
}).mount('#app');
