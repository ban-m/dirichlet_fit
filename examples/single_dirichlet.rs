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
    let mut estim = dirichlet_fit::fit(&data);
    let loss: f64 = data
        .iter()
        .map(|xs| dirichlet_fit::dirichlet_log(xs, param))
        .sum();
    eprintln!("OPT\t{:?}", loss);
    eprintln!("ANS\t{:?}", &[1.0, 8.0, 5.0, 10.0]);
    let fst = estim[0];
    estim.iter_mut().for_each(|x| *x /= fst);
    eprintln!("PRD\t{:?}", estim);
}
