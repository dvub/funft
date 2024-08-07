use std::f32::consts::PI;

// TODO:
// rewrite ALL OF this dogshit code
// BECAUSE JESUS FUCK!
use fundsp::hacker::*;

const OVERSAMPLE_FACTOR: usize = 4;

pub fn generate_graph(
    slow_shared: &Shared,
    fast_shared: &Shared,
    dry_wet: &Shared,
) -> Box<dyn AudioUnit> {
    // The window length, which must be a power of two and at least four,
    // determines the frequency resolution. Latency is equal to the window length.
    let window_length = 2048;
    // hmm...
    let frequencies = generate_frequencies();

    let mixdown = mul(0.5) + mul(0.5);

    // TODO:
    // i want to parameterize these.. but how
    let slow =
        mixdown.clone() >> afollow(0.1, 0.03) >> monitor(slow_shared, Meter::Sample) >> sink();
    let fast =
        mixdown.clone() >> afollow(0.005, 0.1) >> monitor(fast_shared, Meter::Sample) >> sink();

        

    let synth = resynth::<U2, U2, _>(window_length, move |fft| {
        process(fft, &frequencies);
    });

    let wet = var(dry_wet) | var(dry_wet);
    let dry = (1.0 - var(dry_wet)) | (1.0 - var(dry_wet));

    let mixed = (wet * synth) & (dry * multipass::<U2>());

    // now, we may describe the flow of our
    let graph = fast ^ slow ^ mixed;
    Box::new(graph)
}

pub fn process(fft: &mut FftWindow, in_key_frequencies: &[f32]) {
    // println!("{transient}");

    let window_length = fft.length();
    let step_size = window_length / OVERSAMPLE_FACTOR;
    let expected_phase = 2.0 * PI * step_size as f32 / window_length as f32;
    let freq_per_bin = fft.sample_rate() as f32 / window_length as f32;

    for channel in 0..=1 {
        let mut last_phase = vec![0.0; window_length];
        let mut phase_sum = vec![0.0; window_length];

        let mut analysis = vec![(0.0, 0.0); fft.bins()];
        let mut synthesis = vec![(0.0, 0.0); fft.bins()];
        /* this is the analysis step */
        for k in 0..fft.bins() {
            /* compute magnitude and phase */
            let current = fft.at(channel, k);
            let phase = current.atan().re;

            /* compute phase difference */
            let mut tmp = phase - last_phase[k];
            last_phase[k] = phase;

            /* subtract expected phase difference */
            // (from overlap)
            tmp -= k as f32 * expected_phase;

            /* map delta phase into +/- Pi interval */
            let mut qpd = (tmp / PI) as i32;
            let v = qpd & 1;
            if qpd >= 0 {
                qpd += v;
            } else {
                qpd -= v;
            }
            tmp -= PI * qpd as f32;

            /* get deviation from bin frequency from the +/- Pi interval */
            tmp = OVERSAMPLE_FACTOR as f32 * tmp / (2.0 * PI);
            /* compute the k-th partials' true frequency */
            let true_freq = k as f32 * freq_per_bin + tmp * freq_per_bin;
            let magnitude = current.norm();
            /* store magnitude and true frequency in analysis array */
            analysis[k] = (true_freq, magnitude);
        }
        /* this does the actual pitch shifting */
        let mut vov: Vec<Vec<f32>> = vec![Vec::new(); fft.bins()];
        for k in 0..fft.bins() {
            let freq = analysis[k].0;
            let mag = analysis[k].1;

            synthesis[k].0 = freq;

            if is_in_key(in_key_frequencies, freq) {
                synthesis[k].1 += mag;
            } else {
                let mut nearest_ikf = 0.0;
                let mut min_diff = f32::MAX;

                for ikf in in_key_frequencies {
                    let new_diff = (freq - *ikf).abs();
                    if new_diff < min_diff {
                        min_diff = new_diff;
                        nearest_ikf = *ikf;
                    }
                }

                let mut nearest_in_key_bin_index = 0;
                let mut min = f32::MAX;
                for j in 0..fft.bins() {
                    let diff = (nearest_ikf - fft.frequency(j)).abs();
                    if diff < min {
                        min = diff;
                        nearest_in_key_bin_index = j;
                    }
                }

                // this subtraction doesn't work - why?
                synthesis[k].1 += mag * 0.1;

                vov[nearest_in_key_bin_index].push(mag);
            }
        }
        for (i, v) in vov.iter().enumerate() {
            let sum = v.iter().sum::<f32>();
            let len = v.len();
            let avg = sum / max(len, 1) as f32;
            synthesis[i].1 += (avg * 1.25);
        }

        for k in 0..fft.bins() {
            /* get magnitude and true frequency from synthesis array */
            let mag = synthesis[k].1;
            let mut freq = synthesis[k].0;
            /* subtract bin middle frequency */
            freq -= k as f32 * freq_per_bin;
            /* get bin deviation from freq deviation */
            freq /= freq_per_bin;
            /* take oversampling into account */
            freq = 2.0 * PI * freq / OVERSAMPLE_FACTOR as f32;
            /* add the overlap phase advance back in */
            freq += k as f32 * expected_phase;
            /* accumulate delta phase to get bin phase */
            phase_sum[k] += freq;
            let phase = phase_sum[k];
            /* get real and imag part.. */
            let real_component = mag * phase.sin();
            let imaginary_component = mag * phase.cos();

            let output = Complex32::new(real_component, imaginary_component);
            let current = fft.frequency(k);

            let band_min = 0.0;
            let band_max = 44_100.0;

            if (band_min..=band_max).contains(&current) {
                fft.set(channel, k, output);
            }
        }
    }
}

fn is_in_key(in_key_frequencies: &[f32], frequency: f32) -> bool {
    let mut result = false;
    for ikf in in_key_frequencies {
        // TODO:
        // tweak this tolerance!
        if (frequency - *ikf).abs() <= 25.0 {
            result = true;
            break;
        }
    }
    result
}

fn midi_to_freq(x: f32) -> f32 {
    440.0 * 2.0_f32.powf((x - 69.0) / 12.0)
}

#[allow(dead_code)]
fn freq_to_midi(f: f32) -> f32 {
    12.0 * (f / 440.0).log2() + 69.0
}

pub fn generate_frequencies() -> Vec<f32> {
    let max_frequency = 44100.0; // Maximum frequency

    // C Major scale intervals (C, D, E, F, G, A, B, )
    let scale_intervals = vec![0, 2, 3, 5, 7, 10];

    let mut freqs = Vec::new();
    let mut octave = 0;
    let mut base_note = 0;

    // Generate notes until the maximum frequency is reached
    while midi_to_freq(base_note as f32) <= max_frequency {
        for interval in &scale_intervals {
            let midi_note = base_note + interval;
            let f = midi_to_freq(midi_note as f32);
            if f <= max_frequency {
                freqs.push(f);
            }
        }
        octave += 1;
        base_note = octave * 12;
    }

    // Generate frequencies for each note in the scale
    freqs
}

mod tests {

    #[test]
    fn test_conversion() {
        use super::{freq_to_midi, midi_to_freq};
        assert_eq!(midi_to_freq(69.0), 440.0);
        assert_eq!(freq_to_midi(440.0), 69.0);
    }
    #[test]
    fn test_generate_frequencies() {
        use super::generate_frequencies;
        println!("{:?}", generate_frequencies());
    }
}
