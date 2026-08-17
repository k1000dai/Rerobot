//! Hugging Face Hub dataset resolution and snapshot safety.

use rerobot_train::hub::{
    cache_root, is_complete_dataset, validate_repo_id, HubDownloader, HubSnapshot,
};
use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::Path;

#[test]
fn repo_ids_accept_normal_dataset_names_and_reject_path_traversal() {
    assert!(validate_repo_id("lerobot/pusht").is_ok());
    assert!(validate_repo_id("org_name/dataset-name.v3").is_ok());
    for invalid in [
        "",
        "lerobot",
        "/absolute",
        "../escape",
        "org/../escape",
        "org/a\\b",
    ] {
        assert!(validate_repo_id(invalid).is_err(), "accepted {invalid:?}");
    }
}

#[test]
fn an_unqualified_repo_uses_a_revision_safe_cache_root() {
    let root = cache_root(Path::new("/tmp/hf-lerobot"), "lerobot/pusht", "v3.0").unwrap();
    assert_eq!(
        root,
        Path::new("/tmp/hf-lerobot/hub/datasets--lerobot--pusht/snapshots/v3.0")
    );
}

#[test]
fn a_snapshot_is_complete_only_when_the_reader_entrypoint_exists() {
    let dir = tempfile_dir("hub-complete");
    assert!(!is_complete_dataset(&dir));
    std::fs::create_dir_all(dir.join("meta")).unwrap();
    std::fs::write(dir.join("meta/info.json"), "{}").unwrap();
    assert!(!is_complete_dataset(&dir));
    std::fs::create_dir_all(dir.join("meta/episodes/chunk-000")).unwrap();
    std::fs::write(dir.join("meta/tasks.parquet"), b"fixture").unwrap();
    std::fs::write(
        dir.join("meta/episodes/chunk-000/file-000.parquet"),
        b"fixture",
    )
    .unwrap();
    std::fs::create_dir_all(dir.join("data/chunk-000")).unwrap();
    assert!(!is_complete_dataset(&dir));
    std::fs::write(dir.join("data/chunk-000/file-000.parquet"), b"fixture").unwrap();
    assert!(is_complete_dataset(&dir));
}

#[test]
fn snapshot_paths_cannot_escape_the_staging_directory() {
    assert!(HubSnapshot::relative_path("meta/info.json").is_ok());
    for path in ["../outside", "/absolute", "data/../../outside", ""] {
        assert!(
            HubSnapshot::relative_path(path).is_err(),
            "accepted {path:?}"
        );
    }
}

#[test]
fn hub_snapshot_downloads_files_and_publishes_only_after_completion() {
    let tree = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ];
    let (base_url, server) = mock_hub(&tree, false);
    let root = tempfile_dir("hub-download").join("snapshot");
    let result = HubDownloader::new(&base_url).download("org/dataset", &root, "v3.0");
    assert_eq!(result.unwrap(), root);
    assert!(is_complete_dataset(&root));
    assert!(std::fs::read_dir(root.parent().unwrap())
        .unwrap()
        .flatten()
        .all(|entry| !entry.file_name().to_string_lossy().contains("staging")));
    server.join().unwrap();
}

#[test]
fn failed_file_download_does_not_publish_or_leave_staging() {
    let tree = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ];
    let (base_url, server) = mock_hub(&tree, true);
    let parent = tempfile_dir("hub-failure");
    let root = parent.join("snapshot");
    let error = HubDownloader::new(&base_url)
        .download("org/dataset", &root, "v3.0")
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("Hugging Face file request failed"));
    assert!(!root.exists());
    assert!(std::fs::read_dir(parent)
        .unwrap()
        .flatten()
        .all(|entry| !entry.file_name().to_string_lossy().contains("staging")));
    server.join().unwrap();
}

#[test]
fn an_existing_empty_destination_is_rejected_without_being_removed() {
    let parent = tempfile_dir("hub-existing-empty");
    let root = parent.join("snapshot");
    std::fs::create_dir(&root).unwrap();

    let error = HubDownloader::new("http://127.0.0.1:1")
        .download("org/dataset", &root, "v3.0")
        .expect_err("an existing destination must not be overwritten");

    assert!(error.to_string().contains("already exists"));
    assert!(root.is_dir(), "the existing destination was removed");
    assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn a_symlink_destination_is_rejected_even_when_its_target_is_complete() {
    let tree = [
        "meta/info.json",
        "meta/tasks.parquet",
        "meta/episodes/chunk-000/file-000.parquet",
        "data/chunk-000/file-000.parquet",
    ];
    let (base_url, server) = mock_hub(&tree, false);
    let parent = tempfile_dir("hub-symlink-destination");
    let source = parent.join("source");
    HubDownloader::new(&base_url)
        .download("org/dataset", &source, "v3.0")
        .expect("the complete source snapshot downloads");
    server.join().unwrap();

    let alias = parent.join("alias");
    std::os::unix::fs::symlink(&source, &alias).unwrap();
    let error = HubDownloader::new("http://127.0.0.1:1")
        .download("org/dataset", &alias, "v3.0")
        .expect_err("a symlink must not be accepted as a destination alias");

    assert!(error.to_string().contains("destination"));
    assert!(std::fs::symlink_metadata(&alias)
        .unwrap()
        .file_type()
        .is_symlink());
}

#[test]
fn malicious_tree_entry_is_rejected_before_any_file_is_written() {
    let (base_url, server) = mock_hub(&["../outside"], false);
    let parent = tempfile_dir("hub-path");
    let root = parent.join("snapshot");
    assert!(HubDownloader::new(&base_url)
        .download("org/dataset", &root, "v3.0")
        .is_err());
    assert!(!root.exists());
    server.join().unwrap();
}

fn mock_hub(tree: &[&str], fail_info: bool) -> (String, std::thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let tree_json = format!(
        "[{}]",
        tree.iter()
            .map(|path| format!(r#"{{"type":"file","path":"{path}"}}"#))
            .collect::<Vec<_>>()
            .join(",")
    );
    let paths = tree
        .iter()
        .map(|path| (*path).to_owned())
        .collect::<Vec<_>>();
    let expected_requests = if fail_info {
        2
    } else if paths.len() == 1 && paths[0].starts_with("..") {
        1
    } else {
        paths.len() + 1
    };
    let server = std::thread::spawn(move || {
        for _ in 0..expected_requests {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = Vec::new();
            let mut byte = [0_u8; 1];
            while !request.ends_with(b"\r\n\r\n") {
                stream.read_exact(&mut byte).unwrap();
                request.push(byte[0]);
            }
            let request_line = String::from_utf8_lossy(&request);
            let target = request_line.split_whitespace().nth(1).unwrap_or_default();
            let (status, body) = if target.starts_with("/api/datasets/") {
                ("200 OK", tree_json.as_bytes().to_vec())
            } else if fail_info && target.ends_with("/resolve/v3.0/meta/info.json") {
                ("404 Not Found", b"missing".to_vec())
            } else {
                ("200 OK", b"fixture".to_vec())
            };
            let response = format!(
                "HTTP/1.1 {status}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    (format!("http://{address}"), server)
}

fn tempfile_dir(label: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("rerobot-{label}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).unwrap();
    path
}
