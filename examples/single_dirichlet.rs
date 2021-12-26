use dirichlet_fit::Optimizer;
use rand::SeedableRng;
use rand_distr::{Dirichlet, Distribution};
use rand_xoshiro::Xoroshiro128StarStar;
fn main() {
    env_logger::init();
    let seed = 923840;
    let mut rng: Xoroshiro128StarStar = SeedableRng::seed_from_u64(seed);
    let param = &[1.0, 8.0, 5.0, 10.0];
    let dirichlet = Dirichlet::new(param).unwrap();
    let data: Vec<Vec<f64>> = (0..200)
        .map(|_| {
            let mut xs: Vec<f64> = dirichlet.sample(&mut rng);
            xs.iter_mut().for_each(|x| *x = x.ln());
            xs
        })
        .collect();
    let loss: f64 = data
        .iter()
        .map(|xs| dirichlet_fit::dirichlet_log(xs, param))
        .sum();
    eprintln!("OPT\t{:?}", loss);
    let mut estim = {
        let total = data.len() as f64;
        let mut sums = vec![0f64; data[0].len()];
        for xs in data.iter() {
            sums.iter_mut().zip(xs.iter()).for_each(|(s, x)| *s += x);
        }
        sums.iter_mut().for_each(|x| *x /= total);
        let initial_param = [4f64.recip(); 4];
        let initial_param: Vec<_> = initial_param.iter().map(|x| x * 2f64).collect();
        // let mut optimizer = dirichlet_fit::GreedyOptimizer::new(4).set_norm(2f64);
        let mut optimizer = dirichlet_fit::GreedyOptimizer::new(4).set_threshold(0.000000000001); //.set_norm(2f64);
        dirichlet_fit::fit_data_with(&sums, &mut optimizer, &initial_param)
    };
    let loss: f64 = data
        .iter()
        .map(|xs| dirichlet_fit::dirichlet_log(xs, &estim))
        .sum();
    eprintln!("PRED\t{:?}", loss);
    estim.sort_by(|x, y| x.partial_cmp(&y).unwrap());
    let fst = estim[0];
    estim.iter_mut().for_each(|x| *x /= fst);
    eprintln!("PRD\t{:?}", estim);
}
