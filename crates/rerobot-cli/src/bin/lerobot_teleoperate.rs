//! Hardware-gated SO-101 direct-control entry point.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty()
        || args
            .iter()
            .any(|argument| matches!(argument.as_str(), "--help" | "-h" | "--version" | "-V"))
    {
        rerobot_cli::run("lerobot-teleoperate");
    }
    let config = match rerobot_cli::so101::parse(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("lerobot-teleoperate: {error}");
            std::process::exit(rerobot_cli::EXIT_USAGE);
        }
    };
    match rerobot_cli::so101::execute(&config) {
        Ok(line) => println!("{line}"),
        Err(error) => {
            eprintln!("lerobot-teleoperate: {error}");
            std::process::exit(rerobot_cli::EXIT_UNSUPPORTED);
        }
    }
}
