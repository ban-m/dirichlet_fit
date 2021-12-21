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
    let weights = vec![(1f64, vec![1f64; data.len()])];
    let data = [data];
    let initial_param = [4f64.recip(); 4];
    {
        let mut optimizer = dirichlet_fit::AdamOptimizer::new(4);
        let mut estim =
            dirichlet_fit::fit_multiple_with(&data, &weights, &mut optimizer, &initial_param);

        eprintln!("ANS\t{:?}", &[1.0, 8.0, 5.0, 10.0]);
        estim.sort_by(|x, y| x.partial_cmp(&y).unwrap());
        let fst = estim[0];
        estim.iter_mut().for_each(|x| *x /= fst);
        eprintln!("PRD\t{:?}", estim);
    }
    {
        let initial_param: Vec<_> = initial_param.iter().map(|x| x * 2f64).collect();
        let mut optimizer = dirichlet_fit::GreedyOptimizer::new(4).set_norm(2f64);
        let mut estim =
            dirichlet_fit::fit_multiple_with(&data, &weights, &mut optimizer, &initial_param);
        estim.sort_by(|x, y| x.partial_cmp(&y).unwrap());
        let fst = estim[0];
        estim.iter_mut().for_each(|x| *x /= fst);
        eprintln!("PRD\t{:?}", estim);
    }
}
