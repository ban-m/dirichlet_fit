use dirichlet_fit::Dirichlet;
fn main() {
    env_logger::init();
    let args: Vec<_> = std::env::args().collect();
    use std::io::{BufRead, BufReader};
    for line in std::fs::File::open(&args[1])
        .map(BufReader::new)
        .unwrap()
        .lines()
        .filter_map(|x| x.ok())
        .filter(|line| line.starts_with('['))
    // .skip(6)
    // .take(1)
    {
        let mut line = line.split('\t');
        let param: Vec<f64> = line
            .next()
            .unwrap()
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|x| x.trim().parse().unwrap())
            .collect();
        let data: Vec<f64> = line
            .next()
            .unwrap()
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(|x| x.trim().parse().unwrap())
            .collect();
        //    eprintln!("{:?}", data);
        let data = sanitize(data);
        // eprintln!("{:?}", data);
        let param = estim(&data);
        let mut dirichlet = Dirichlet::new(&param);
        let start = std::time::Instant::now();
        dirichlet.update(&data, Some(2f64));
        let end = std::time::Instant::now();
        let duration = (end - start).as_micros();
        eprintln!(
            "{}\t{}\t{}\t{}",
            duration,
            dirichlet,
            vec2str(&param),
            vec2str(&data)
        );
    }
}

fn estim(xs: &[f64]) -> Vec<f64> {
    let mut param: Vec<_> = xs
        .iter()
        .map(|&x| {
            if -2.2 < x {
                x.exp() + 0.5
            } else {
                -(x - dirichlet_fit::DIGAM_ONE).recip()
            }
        })
        .collect();
    let sum: f64 = param.iter().sum();
    param.iter_mut().for_each(|x| *x = *x * 2f64 / sum);
    param
}

fn vec2str(xs: &[f64]) -> String {
    let xs: Vec<_> = xs.iter().map(|&x| format!("{:6.1}", x)).collect();
    xs.join(",")
}

// e^-30 IS zero.
const SMALL_LOG: f64 = -30f64;
fn sanitize(mut data: Vec<f64>) -> Vec<f64> {
    let dim = data.len() as f64;
    let ofs = SMALL_LOG - dim.ln();
    data = data
        .into_iter()
        .map(|x| {
            if x < ofs {
                ofs + ((x - ofs).exp() + 1f64).ln()
            } else {
                x + (1f64 + (ofs - x).exp()).ln()
            }
        })
        .collect();
    data
}
