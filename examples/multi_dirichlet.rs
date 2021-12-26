use rand::SeedableRng;
use rand_distr::{Dirichlet, Distribution};
use rand_xoshiro::Xoroshiro128StarStar;
fn main() {
    // Multiple dirichlet distribution.
    env_logger::init();
    let seed = 923840;
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(seed);
    let params = vec![vec![2.0, 1.0, 2.0, 1.0], vec![1.0, 8.0, 5.0, 10.0]];
    let dirichlets: Vec<_> = params
        .iter()
        .map(|param| Dirichlet::new(param).unwrap())
        .collect();
    let datalen = 200;
    let batch_size = 3;
    let (batch, weights): (Vec<_>, Vec<_>) = (0..batch_size)
        .map(|bt| {
            let (conf, ok, err) = match bt {
                0 => (0.01, 0.8, 0.3),
                1 => (0.05, 0.9, 0.2),
                2 => (0.9, 1f64, 0.01),
                _ => panic!(),
            };
            let (data, weights): (Vec<_>, Vec<_>) = (0..datalen)
                .map(|i| {
                    let dir = params.len() * i / datalen;
                    let mut xs: Vec<f64> = dirichlets[dir].sample(&mut rng);
                    xs.iter_mut().for_each(|x| *x = x.ln());
                    match dir {
                        0 => (xs, err),
                        1 => (xs, ok),
                        _ => panic!(),
                    }
                })
                .unzip();
            (data, (conf, weights))
        })
        .unzip();
    let (mut sums, mut total) = (vec![0f64; params[0].len()], 0f64);
    for (data, (w_data, ws)) in batch.iter().zip(weights.iter()) {
        total += w_data * ws.iter().sum::<f64>();
        for (w, xs) in ws.iter().zip(data.iter()) {
            sums.iter_mut()
                .zip(xs.iter())
                .for_each(|(s, x)| *s += w_data * w * x);
        }
    }
    sums.iter_mut().for_each(|x| *x /= total);
    let mut estim = dirichlet_fit::fit_data(&sums);
    estim.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let norm = estim[0];
    estim.iter_mut().for_each(|x| *x /= norm);
    eprintln!("{:?}", estim);
}
