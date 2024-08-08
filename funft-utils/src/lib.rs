use fundsp::hacker::*;
use std::{cmp::Ordering, f32::consts::PI};

const OVERSAMPLE_FACTOR: usize = 4;

// TODO:
// rewrite ALL OF this dogshit code

pub fn generate_graph(
    slow_shared: &Shared,
    fast_shared: &Shared,
    dry_wet: &Shared,
    intervals: Vec<Shared>,
) -> Box<dyn AudioUnit> {
    // The window length, which must be a power of two and at least four,
    // determines the frequency resolution.
    // **Latency is equal to the window length.**
    let window_length = 2048;

    let mixdown = mul(0.5) + mul(0.5);

    // gives us nice dips when transients occur

    // 0.25, 0.03
    // 0.005, 0.1

    // turning up the first value will overall increase the wetness of the effect
    // 0.5 is a bit too much for first value

    // second value has a similar effect
    // 0.5 or 0.6 is probably a good limit

    let slow =
        mixdown.clone() >> afollow(0.25, 0.015) >> monitor(slow_shared, Meter::Sample) >> sink();
    // this fast envelope follow should not be tweaked too much, if at all
    let fast =
        mixdown.clone() >> afollow(0.005, 0.1) >> monitor(fast_shared, Meter::Sample) >> sink();

    // alternative idea: use a peak meter
    // mixdown >> monitor(&dry_wet, Meter::Peak(0.4)) >> sink()

    let synth = resynth::<U2, U2, _>(window_length, move |fft| {
        process(fft, &intervals);
    });

    let wet = var(dry_wet) | var(dry_wet);
    let dry = (1.0 - var(dry_wet)) | (1.0 - var(dry_wet));

    let mixed = (wet * synth) & (dry * multipass::<U2>());

    // now that we've described the components, we can put them together
    let graph = fast ^ slow ^ mixed;
    Box::new(graph)
}

pub fn process(fft: &mut FftWindow, intervals: &[Shared]) {
    let intervals: Vec<_> = intervals.iter().map(|x| x.value()).collect();
    let in_key_frequencies = &generate_frequencies(&intervals);

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

        // TODO:
        // parameterize!
        let processing_band_min = 100.0;
        let processing_band_max = 10_000.0;

        // https://blogs.zynaptiq.com/bernsee/pitch-shifting-using-the-ft/

        // most of this code is translated from this example (c++ to rust)
        // TODO:
        // review to ensure no translation mistakes

        // https://blogs.zynaptiq.com/bernsee/repo/smbPitchShift.cpp

        /* this is the analysis step */
        for k in 0..fft.bins() {
            if !(processing_band_min..=processing_band_max).contains(&fft.frequency(k)) {
                continue;
            }

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
        let mut vov: Vec<(Vec<f32>, f32)> = vec![(Vec::new(), 0.0); fft.bins()];
        for k in 0..fft.bins() {
            if !(processing_band_min..=processing_band_max).contains(&fft.frequency(k)) {
                continue;
            }

            let freq = analysis[k].0;
            let mag = analysis[k].1;

            synthesis[k].0 = freq;

            if is_in_key(in_key_frequencies, freq) {
                synthesis[k].1 += mag;
            } else {
                // TODO:
                // rework this entire stupid algorithm
                let mut nearest = (0.0, 0.0);
                let mut min_diff = f32::MAX;

                for ikf in in_key_frequencies {
                    let new_diff = (freq - ikf.0).abs();
                    if new_diff < min_diff {
                        min_diff = new_diff;
                        nearest = *ikf;
                    }
                }

                let mut nearest_in_key_bin_index = 0;
                let mut min = f32::MAX;
                for j in 0..fft.bins() {
                    let diff = (nearest.0 - fft.frequency(j)).abs();
                    if diff < min {
                        min = diff;
                        nearest_in_key_bin_index = j;
                    }
                }
                // TODO:
                // parameterize
                let amplitude = (k as f32).log10() * 0.075;
                synthesis[k].1 += mag * amplitude;

                let weight = nearest.1;

                vov[nearest_in_key_bin_index].0.push(mag);
                vov[nearest_in_key_bin_index].1 = weight;
            }
        }
        for (i, (v, weight)) in vov.iter().enumerate() {
            // TODO:
            // please rewrite this better.
            let sum = v.iter().sum::<f32>();
            let len = v.len();
            let avg = sum / max(len, 1) as f32;

            // TODO:
            // fix this if statement to prevent NaN LOLLL
            // parameterize this multiplier at the end!
            // figure out other stuff other than log, maybe parameterize that too!
            if i > 0 {
                let amplitude = (i as f32).log10() * 2.0 * weight;
                // println!("{}", amplitude);
                synthesis[i].1 += avg * amplitude;
            }
        }

        for k in 0..fft.bins() {
            if !(processing_band_min..=processing_band_max).contains(&fft.frequency(k)) {
                fft.set(channel, k, fft.at(channel, k));
                continue;
            }

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

fn is_in_key(in_key_frequencies: &[(f32, f32)], frequency: f32) -> bool {
    // would there be a way to optimize this?
    let mut result = false;
    for ikf in in_key_frequencies {
        // TODO:
        // tweak this tolerance!
        if (frequency - ikf.0).abs() <= 25.0 {
            result = true;
            break;
        }
    }
    result
}

// TODO:
// rewrite this
pub fn generate_frequencies(intervals: &[f32]) -> Vec<(f32, f32)> {
    // TODO:
    // turn this into a parameter of the function, being sample rate
    let sample_rate = 44_100.0;
    let max_frequency = sample_rate / 2.0; // Maximum frequency

    // convert our incoming array into an array of tuples
    let mut is = Vec::new();
    for (interval, velocity) in intervals.iter().enumerate() {
        is.push((interval, *velocity));
    }

    let mut freqs = Vec::new();

    // temporary variables
    let mut octave = 0;
    let mut base_note = 0;

    // TODO:
    // there is definitely some optimizations to be had here

    // also, good LORD please fix the indents
    while midi_to_freq(base_note as f32) <= max_frequency {
        for (interval, velocity) in &is {
            let midi_note = base_note + interval;
            if *velocity > 0.0 {
                let f = midi_to_freq(midi_note as f32);
                if f <= max_frequency {
                    // 2 is a good number for harmonics, at least in my opinion
                    // TODO:
                    // parameterize!

                    let num_harmonics = 2;
                    // note here that the 0th harmonic is the fundamental frequency
                    for nth_harmonic in 0..=num_harmonics {
                        // TODO:
                        // do something cool here
                        // let harmonic_weight = 1.0 - (nth_harmonic as f32 / num_harmonics as f32);
                        let harmonic_weight = 1.0;
                        let frequency = f * nth_harmonic as f32;

                        if frequency < max_frequency {
                            let tuple = (frequency, *velocity * harmonic_weight);
                            freqs.push(tuple);
                        }
                    }
                }
            }
        }
        octave += 1;
        base_note = octave * 12;
    }

    // TODO:
    // figure out how you want to sort??
    freqs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    freqs.dedup();

    freqs
}

fn midi_to_freq(x: f32) -> f32 {
    440.0 * 2.0_f32.powf((x - 69.0) / 12.0)
}

// i kept this here because uh.. idk i might use it later?
#[allow(dead_code)]
fn freq_to_midi(f: f32) -> f32 {
    12.0 * (f / 440.0).log2() + 69.0
}

mod tests {

    #[test]
    fn test_conversion() {
        use super::{freq_to_midi, midi_to_freq};
        assert_eq!(midi_to_freq(69.0), 440.0);
        assert_eq!(freq_to_midi(440.0), 69.0);
    }
}
