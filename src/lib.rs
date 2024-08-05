use core::f32;
use fundsp::hacker::*;
use funft_utils::{find_adjacent_indices, generate_frequencies};
use nih_plug::prelude::*;

use std::sync::Arc;
use typenum::{UInt, UTerm};

#[derive(Params)]

struct GainParams {}

struct Gain {
    graph: Box<dyn AudioUnit>,
    input_buffer: BufferArray<UInt<UInt<UTerm, typenum::B1>, typenum::B0>>,
    output_buffer: BufferArray<UInt<UInt<UTerm, typenum::B1>, typenum::B0>>,
    params: Arc<GainParams>,
}

#[derive(PartialEq, nih_plug::prelude::Enum)]
pub enum LevelDetection {
    Rms,
    Peak,
}

impl Default for Gain {
    fn default() -> Self {
        // The window length, which must be a power of two and at least four,
        // determines the frequency resolution. Latency is equal to the window length.
        let window_length = 512;
        let frequencies = generate_frequencies();

        let synth = resynth::<U2, U2, _>(window_length, move |fft| {
            for channel in 0..=1 {
                // iterate through fft bins
                for i in 0..fft.bins() {
                    let current_frequency = fft.frequency(i);

                    if let Some((l, r)) = find_adjacent_indices(&frequencies, current_frequency) {
                        let midpoint = (frequencies[l] + frequencies[r]) / 2.0;

                        let half = (midpoint - l as f32).abs();

                        let diff = (current_frequency - midpoint).abs();

                        let amp = (diff / half).powi(2);
                        fft.set(channel, i, fft.at(channel, i) * amp);
                    }
                }
            }
        });

        let graph = synth;

        Self {
            graph: Box::new(graph),
            params: Arc::new(GainParams {}),

            input_buffer: BufferArray::<U2>::new(),
            output_buffer: BufferArray::<U2>::new(),
        }
    }
}

impl Plugin for Gain {
    const NAME: &'static str = "Gain";
    const VENDOR: &'static str = "Moist Plugins GmbH";
    // You can use `env!("CARGO_PKG_HOMEPAGE")` to reference the homepage field from the
    // `Cargo.toml` file here
    const URL: &'static str = "https://youtu.be/dQw4w9WgXcQ";
    const EMAIL: &'static str = "info@example.com";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    // The first audio IO layout is used as the default. The other layouts may be selected either
    // explicitly or automatically by the host or the user depending on the plugin API/backend.
    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),

            aux_input_ports: &[],
            aux_output_ports: &[],

            // Individual ports and the layout as a whole can be named here. By default these names
            // are generated as needed. This layout will be called 'Stereo', while the other one is
            // given the name 'Mono' based no the number of input and output channels.
            names: PortNames::const_default(),
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    const MIDI_INPUT: MidiConfig = MidiConfig::None;
    // Setting this to `true` will tell the wrapper to split the buffer up into smaller blocks
    // whenever there are inter-buffer parameter changes. This way no changes to the plugin are
    // required to support sample accurate automation and the wrapper handles all of the boring
    // stuff like making sure transport and other timing information stays consistent between the
    // splits.
    const SAMPLE_ACCURATE_AUTOMATION: bool = true;

    // If the plugin can send or receive SysEx messages, it can define a type to wrap around those
    // messages here. The type implements the `SysExMessage` trait, which allows conversion to and
    // from plain byte buffers.
    type SysExMessage = ();
    // More advanced plugins can use this to run expensive background tasks. See the field's
    // documentation for more information. `()` means that the plugin does not have any background
    // tasks.
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // TODO:
        // use BigBlockAdapter

        // offset is the sample offset from beginning of buffer,
        // we dont care about that here
        for (_offset, mut block) in buffer.iter_blocks(MAX_BUFFER_SIZE) {
            // write into input buffer
            for (sample_index, mut channel_samples) in block.iter_samples().enumerate() {
                for channel_index in 0..=1 {
                    let sample = *channel_samples.get_mut(channel_index).unwrap();
                    self.input_buffer
                        .buffer_mut()
                        .set_f32(channel_index, sample_index, sample);
                }
            }

            self.graph.process(
                block.samples(),
                &self.input_buffer.buffer_ref(),
                &mut self.output_buffer.buffer_mut(),
            );

            // write from output buffer
            for (index, mut channel_samples) in block.iter_samples().enumerate() {
                for n in 0..=1 {
                    let sample_from_buf = self.output_buffer.buffer_ref().at_f32(n, index);
                    *channel_samples.get_mut(n).unwrap() = sample_from_buf;
                }
            }
        }

        ProcessStatus::Normal
    }

    // This can be used for cleaning up special resources like socket connections whenever the
    // plugin is deactivated. Most plugins won't need to do anything here.
    fn deactivate(&mut self) {}
}

impl ClapPlugin for Gain {
    const CLAP_ID: &'static str = "com.moist-plugins-gmbh.gain";
    const CLAP_DESCRIPTION: Option<&'static str> = Some("A smoothed gain parameter example plugin");
    const CLAP_MANUAL_URL: Option<&'static str> = Some(Self::URL);
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for Gain {
    const VST3_CLASS_ID: [u8; 16] = *b"GainMoistestPlug";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nih_export_clap!(Gain);
nih_export_vst3!(Gain);
