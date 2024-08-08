use fundsp::prelude::*;

use crate::process;

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
