//! Hardware-gated SO-101 motor discovery entry point.

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty()
        || args
            .iter()
            .any(|argument| matches!(argument.as_str(), "--help" | "-h" | "--version" | "-V"))
    {
        rerobot_cli::run("lerobot-setup-motors");
    }
    if !args
        .iter()
        .any(|argument| argument.starts_with("--robot.type="))
    {
        args.push("--robot.type=so101_follower".to_owned());
    }
    if !args
        .iter()
        .any(|argument| argument.starts_with("--robot.action="))
    {
        args.push("--robot.action=ping".to_owned());
    }
    let config = match rerobot_cli::so101::parse(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("lerobot-setup-motors: {error}");
            std::process::exit(rerobot_cli::EXIT_USAGE);
        }
    };
    match rerobot_cli::so101::execute(&config) {
        Ok(line) => println!("{line}"),
        Err(error) => {
            eprintln!("lerobot-setup-motors: {error}");
            std::process::exit(rerobot_cli::EXIT_UNSUPPORTED);
        }
    }
}
