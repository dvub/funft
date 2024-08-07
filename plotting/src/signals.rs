use rand::Rng;



pub fn gen_noise() -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut v = Vec::new();
    let amplitude = 1.0;

    let max = 44100;
    for _ in 0..=1000 {
        v.push(0.0);
    }

    for _ in 0..=1 {
        for i in 0..=22_050{
            let mult = 1.0 - (i as f32 / max as f32);
            // println!("{}", mult);
            v.push(rng.gen_range(-amplitude..amplitude) * mult);
        }
    }

    v
    
}
