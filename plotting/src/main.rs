mod plots;
mod signals;
use plots::{frequency_plot, generic_plot};

use fundsp::hacker::*;
use funft_utils::{generate_frequencies, generate_graph};
use numeric_array::{generic_array::arr, NumericArray};

use plotters::prelude::*;
use rand::Rng;
use realfft::RealFftPlanner;
use signals::gen_noise;
use std::{
    fs::create_dir,
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

const OUT_DIR: &str = "./output";
fn main() {
    // generate a test signal to operate on
    let mut noise = gen_noise();
    // save an unprocessed copy
    let mut pre_noise = noise.clone();

    let frequencies = generate_frequencies();

    let lasr_shared = shared(0.0);
    let sasr_shared = shared(0.0);
    let delta = shared(0.0);

    // this graph will handle all of our audio processing
    let mut graph = generate_graph(&lasr_shared, &sasr_shared, &delta);

    // vector to hold our processor's transient detection
    let mut deltas = Vec::new();
    let mut bleh = Vec::new();
    let mut bleh2 = Vec::new();

    for sample in &mut noise {
        // convert our data into something that our graph can work with
        let samples_into_array = &NumericArray::new(arr![*sample; 2]);
        let mut out = [0.0; 2];
        // process
        graph.tick(samples_into_array, &mut out);
        *sample = out[0];

        let lasr = lasr_shared.value().abs();
        let sasr = sasr_shared.value().abs();
        let d = (lasr / sasr).clamp(0.0, 1.0);
        delta.set(d);

        deltas.push(d);
        bleh.push(lasr);
        bleh2.push(sasr);
    }
    // now its time to create our fun plots!
    let time = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis();

    if !Path::new(OUT_DIR).exists() {
        create_dir(OUT_DIR).unwrap();
    }
    let dir = Path::new(OUT_DIR).join(time.to_string());
    create_dir(&dir).unwrap();
    generic_plot(
        &dir.join("time domain.png"),
        vec![
            (pre_noise.clone(), BLUE),
            (deltas, RED),
            (bleh, GREEN),
            (bleh2, BLACK),
        ],
    );

    let mut planner = RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(pre_noise.len());

    let mut output = r2c.make_output_vec();
    r2c.process(&mut pre_noise, &mut output).unwrap();

    let mut output2 = r2c.make_output_vec();
    r2c.process(&mut noise, &mut output2).unwrap();

    let noise_fft_mags: Vec<_> = output.iter().map(|s| s.norm()).collect();
    let processed_fft_mags: Vec<_> = output2.iter().map(|s| s.norm()).collect();

    frequency_plot(
        &dir.join("freq domain.png"),
        vec![(noise_fft_mags, BLUE), (processed_fft_mags, RED)],
        &frequencies,
        44100,
    );
}
