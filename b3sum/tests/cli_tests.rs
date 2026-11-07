use duct::cmd;
use std::ffi::OsString;
use std::fs;
use std::io::prelude::*;
use std::path::PathBuf;

pub fn b3sum_exe() -> PathBuf {
    env!("CARGO_BIN_EXE_b3sum").into()
}

/// A reader that goes away ends b3sum at once and quietly (SIGPIPE), as
/// it ends GNU's tools: more results than a pipe holds, the reader gone
/// after one line.
#[test]
#[cfg(unix)]
fn test_reader_gone() {
    use std::io::BufRead;
    use std::os::unix::process::ExitStatusExt;
    let dir = tempfile::tempdir().unwrap();
    let files: Vec<PathBuf> = (0..2000)
        .map(|i| {
            let path = dir.path().join(format!("{i:04}"));
            fs::write(&path, i.to_string()).unwrap();
            path
        })
        .collect();
    let mut child =
        std::process::Command::new(b3sum_exe()).args(&files).stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::piped()).spawn().unwrap();
    let mut line = String::new();
    std::io::BufReader::new(child.stdout.take().unwrap()).read_line(&mut line).unwrap();
    let output = child.wait_with_output().unwrap();
    const SIGPIPE: i32 = 13; // on Linux and macOS
    assert_eq!(output.status.signal(), Some(SIGPIPE), "{:?}", output.status);
    assert!(output.stderr.is_empty(), "{}", String::from_utf8_lossy(&output.stderr));
}

/// A file whose reported size is not its content (Linux's /proc and /sys
/// report 0 or 4096 bytes) hashes as its content, as other hash tools
/// hash it: b3sum uses a file's size to choose how to read it, never to
/// judge it.
#[test]
#[cfg(target_os = "linux")]
fn test_pseudo_file() {
    blake3::initialize();
    // /proc/version: reported as 0 bytes, its content the same on every read.
    let Ok(content) = fs::read("/proc/version") else { return };
    let output = cmd!(b3sum_exe(), "--no-names", "/proc/version").stdout_capture().stderr_capture().unchecked().run().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    assert_eq!(blake3::convenience::hash(&content).to_hex().to_string(), String::from_utf8_lossy(&output.stdout).trim_end());
}

#[test]
fn test_hash_one() {
    blake3::initialize();
    let expected = format!("{}  -", blake3::convenience::hash(b"foo").to_hex());
    let output = cmd!(b3sum_exe()).stdin_bytes("foo").read().unwrap();
    assert_eq!(&*expected, output);
}

#[test]
fn test_hash_one_tag() {
    blake3::initialize();
    let expected = format!("BLAKE3 (-) = {}", blake3::convenience::hash(b"foo").to_hex());
    let output = cmd!(b3sum_exe(), "--tag").stdin_bytes("foo").read().unwrap();
    assert_eq!(&*expected, output);
}

#[test]
fn test_hash_one_raw() {
    blake3::initialize();
    let expected = blake3::convenience::hash(b"foo").as_bytes().to_owned();
    let output = cmd!(b3sum_exe(), "--raw").stdin_bytes("foo").stdout_capture().run().unwrap().stdout;
    assert_eq!(expected, output.as_slice());
}

#[test]
fn test_hash_many() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    let file1 = dir.path().join("file1");
    fs::write(&file1, b"foo").unwrap();
    let file2 = dir.path().join("file2");
    fs::write(&file2, b"bar").unwrap();

    let output = cmd!(b3sum_exe(), &file1, &file2).read().unwrap();
    let foo_hash = blake3::convenience::hash(b"foo");
    let bar_hash = blake3::convenience::hash(b"bar");
    let expected = format!("{}  {}\n{}  {}", foo_hash.to_hex(), file1.to_string_lossy(), bar_hash.to_hex(), file2.to_string_lossy());
    assert_eq!(expected, output);

    let output_no_names = cmd!(b3sum_exe(), "--no-names", &file1, &file2).read().unwrap();
    let expected_no_names = format!("{}\n{}", foo_hash.to_hex(), bar_hash.to_hex());
    assert_eq!(expected_no_names, output_no_names);
}

/// Many files of many sizes, more than b3sum keeps in flight (on Linux, up
/// to 64 read ahead), each digest the library's and in
/// the order given: empty files, small ones, and ones beside the 512 KiB
/// boundary where mapping takes over.
#[test]
fn test_hash_many_sizes() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    let sizes = [0usize, 1, 64, 1023, 1024, 1025, 16 << 10, (512 << 10) - 1, 512 << 10, (512 << 10) + 1];
    let mut expected = String::new();
    let mut files = Vec::new();
    for i in 0..300 {
        let len = sizes[i % sizes.len()];
        let bytes: Vec<u8> = (0..len).map(|j| (j as u32 * 31 + i as u32) as u8).collect();
        let path = dir.path().join(format!("{i:03}"));
        fs::write(&path, &bytes).unwrap();
        expected += &format!("{}\n", blake3::convenience::hash(&bytes).to_hex());
        files.push(path);
    }
    // One thread holds the deepest batch (64 files), two half of it.
    for threads in [&["--num-threads=1"][..], &["--num-threads=2"], &[]] {
        let args = ["--no-names"].iter().chain(threads).map(Into::into).chain(files.iter().map(|p| p.clone().into_os_string()));
        let output = cmd(b3sum_exe(), args.collect::<Vec<std::ffi::OsString>>()).read().unwrap();
        assert_eq!(expected.trim_end(), output, "{threads:?}");
    }
}

#[test]
fn test_hash_many_tag() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    let file1 = dir.path().join("file1");
    fs::write(&file1, b"foo").unwrap();
    let file2 = dir.path().join("file2");
    fs::write(&file2, b"bar").unwrap();

    let output = cmd!(b3sum_exe(), "--tag", &file1, &file2).read().unwrap();
    let foo_hash = blake3::convenience::hash(b"foo");
    let bar_hash = blake3::convenience::hash(b"bar");
    let expected = format!("BLAKE3 ({}) = {}\nBLAKE3 ({}) = {}", file1.to_string_lossy(), foo_hash.to_hex(), file2.to_string_lossy(), bar_hash.to_hex());
    assert_eq!(expected, output);
}

#[test]
fn test_missing_files() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    let file1 = dir.path().join("file1");
    fs::write(&file1, b"foo").unwrap();
    let file2 = dir.path().join("file2");
    fs::write(&file2, b"bar").unwrap();

    let output = cmd!(b3sum_exe(), "file1", "missing_file", "file2").dir(dir.path()).stdout_capture().stderr_capture().unchecked().run().unwrap();
    assert!(!output.status.success());

    let foo_hash = blake3::convenience::hash(b"foo");
    let bar_hash = blake3::convenience::hash(b"bar");
    let expected_stdout = format!("{}  file1\n{}  file2\n", foo_hash.to_hex(), bar_hash.to_hex());
    assert_eq!(expected_stdout.as_bytes(), &output.stdout[..]);

    let bing_error = fs::File::open(dir.path().join("missing_file")).unwrap_err();
    let expected_stderr = format!("b3sum: missing_file: {}\n", bing_error.to_string());
    assert_eq!(expected_stderr.as_bytes(), &output.stderr[..]);
}

#[test]
fn test_hash_length_and_seek() {
    blake3::initialize();
    let mut expected = [0; 100];
    blake3::convenience::Hasher::new(blake3::Mode::Plain, None).update(b"foo").finalize_xof().fill(&mut expected);
    let output = cmd!(b3sum_exe(), "--raw", "--length=100").stdin_bytes("foo").stdout_capture().run().unwrap().stdout;
    assert_eq!(expected[..], output);

    let short_output = cmd!(b3sum_exe(), "--raw", "--length=99").stdin_bytes("foo").stdout_capture().run().unwrap().stdout;
    assert_eq!(expected[..99], short_output);

    let seek1_output = cmd!(b3sum_exe(), "--raw", "--length=99", "--seek=1").stdin_bytes("foo").stdout_capture().run().unwrap().stdout;
    assert_eq!(expected[1..], seek1_output);

    let seek99_output = cmd!(b3sum_exe(), "--raw", "--length=1", "--seek=99").stdin_bytes("foo").stdout_capture().run().unwrap().stdout;
    assert_eq!(expected[99..], seek99_output);
}

#[test]
fn test_keyed() {
    blake3::initialize();
    let key = [42; blake3::KEY_LEN];
    let f = tempfile::NamedTempFile::new().unwrap();
    f.as_file().write_all(b"foo").unwrap();
    f.as_file().flush().unwrap();
    let expected = blake3::convenience::hash_with(blake3::Mode::Keyed(&key), b"foo").to_hex();
    let output = cmd!(b3sum_exe(), "--keyed", "--no-names", f.path()).stdin_bytes(&key[..]).read().unwrap();
    assert_eq!(&*expected, &*output);

    // Make sure that keys of the wrong length lead to errors.
    for bad_length in [0, 1, blake3::KEY_LEN - 1, blake3::KEY_LEN + 1] {
        dbg!(bad_length);
        let output = cmd!(b3sum_exe(), "--keyed", f.path()).stdin_bytes(vec![0; bad_length]).stdout_capture().stderr_capture().unchecked().run().unwrap();
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        // Make sure the error message is relevant.
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        assert!(stderr.contains("key bytes"));
    }
}

#[test]
fn test_derive_key() {
    blake3::initialize();
    let context = "BLAKE3 2019-12-28 10:28:41 example context";
    let f = tempfile::NamedTempFile::new().unwrap();
    f.as_file().write_all(b"key material").unwrap();
    f.as_file().flush().unwrap();
    let expected = hex::encode(*blake3::convenience::hash_with(blake3::Mode::DeriveKey(context), b"key material").as_bytes());
    let output = cmd!(b3sum_exe(), "--derive-key", context, "--no-names", f.path()).read().unwrap();
    assert_eq!(&*expected, &*output);
}

#[test]
fn test_no_mmap() {
    blake3::initialize();
    let f = tempfile::NamedTempFile::new().unwrap();
    f.as_file().write_all(b"foo").unwrap();
    f.as_file().flush().unwrap();

    let expected = blake3::convenience::hash(b"foo").to_hex();
    let output = cmd!(b3sum_exe(), "--no-mmap", "--no-names", f.path()).read().unwrap();
    assert_eq!(&*expected, &*output);
}

#[test]
fn test_length_without_value_is_an_error() {
    blake3::initialize();
    let result = cmd!(b3sum_exe(), "--length").stdin_bytes("foo").stderr_capture().run();
    assert!(result.is_err());
}

#[test]
fn test_raw_with_multi_files_is_an_error() {
    blake3::initialize();
    let f1 = tempfile::NamedTempFile::new().unwrap();
    let f2 = tempfile::NamedTempFile::new().unwrap();

    // Make sure it doesn't error with just one file
    let result = cmd!(b3sum_exe(), "--raw", f1.path()).stdout_capture().run();
    assert!(result.is_ok());

    // Make sure it errors when both file are passed
    let result = cmd!(b3sum_exe(), "--raw", f1.path(), f2.path()).stderr_capture().run();
    assert!(result.is_err());
}

#[test]
#[cfg(unix)]
fn test_newline_and_backslash_escaping_on_unix() {
    blake3::initialize();
    let empty_hash = blake3::convenience::hash(b"").to_hex();
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir(dir.path().join("subdir")).unwrap();
    let names = ["abcdef", "abc\ndef", "abc\\def", "abc\rdef", "abc\r\ndef", "subdir/foo"];
    let mut paths = Vec::new();
    for name in &names {
        let path = dir.path().join(name);
        println!("creating file at {:?}", path);
        fs::write(&path, b"").unwrap();
        paths.push(path);
    }
    let output = cmd(b3sum_exe(), &names).dir(dir.path()).read().unwrap();
    let expected = format!(
        "\
{0}  abcdef
\\{0}  abc\\ndef
\\{0}  abc\\\\def
\\{0}  abc\\rdef
\\{0}  abc\\r\\ndef
{0}  subdir/foo",
        empty_hash,
    );
    println!("output");
    println!("======");
    println!("{}", output);
    println!();
    println!("expected");
    println!("========");
    println!("{}", expected);
    println!();
    assert_eq!(expected, output);
}

#[test]
#[cfg(unix)]
fn test_invalid_unicode_on_unix() {
    blake3::initialize();
    use std::os::unix::ffi::OsStringExt;

    let empty_hash = blake3::convenience::hash(b"").to_hex();
    let dir = tempfile::tempdir().unwrap();
    let names = ["abcdef".into(), OsString::from_vec(b"abc\xffdef".to_vec())];
    let mut paths = Vec::new();
    for name in &names {
        let path = dir.path().join(name);
        println!("creating file at {:?}", path);
        // Note: Some operating systems, macOS in particular, simply don't
        // allow invalid Unicode in filenames. On those systems, this write
        // will fail. That's fine, we'll just short-circuit this test in that
        // case. But assert that at least Linux allows this.
        let write_result = fs::write(&path, b"");
        if cfg!(target_os = "linux") {
            write_result.expect("Linux should allow invalid Unicode");
        } else if write_result.is_err() {
            return;
        }
        paths.push(path);
    }
    let output = cmd(b3sum_exe(), &names).dir(dir.path()).read().unwrap();
    let expected = format!(
        "\
{0}  abcdef
{0}  abc�def",
        empty_hash,
    );
    println!("output");
    println!("======");
    println!("{}", output);
    println!();
    println!("expected");
    println!("========");
    println!("{}", expected);
    println!();
    assert_eq!(expected, output);
}

#[test]
fn test_check() {
    blake3::initialize();
    // Make a directory full of files, and make sure the b3sum output in that
    // directory is what we expect.
    let a_hash = blake3::convenience::hash(b"a").to_hex();
    let b_hash = blake3::convenience::hash(b"b").to_hex();
    let cd_hash = blake3::convenience::hash(b"cd").to_hex();
    for tagged in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a"), b"a").unwrap();
        fs::write(dir.path().join("b"), b"b").unwrap();
        fs::create_dir(dir.path().join("c")).unwrap();
        fs::write(dir.path().join("c/d"), b"cd").unwrap();
        dbg!(tagged);
        let mut args = vec!["a", "b", "c/d"];
        if tagged {
            args.push("--tag");
        }
        let output = cmd(b3sum_exe(), args).dir(dir.path()).stdout_capture().stderr_capture().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_checkfile = if tagged {
            format!(
                "BLAKE3 (a) = {}\n\
                 BLAKE3 (b) = {}\n\
                 BLAKE3 (c/d) = {}\n",
                a_hash, b_hash, cd_hash,
            )
        } else {
            format!(
                "{}  a\n\
                 {}  b\n\
                 {}  c/d\n",
                a_hash, b_hash, cd_hash,
            )
        };
        dbg!(&expected_checkfile);
        assert_eq!(expected_checkfile, stdout);
        assert_eq!("", stderr);

        // Now use the output we just validated as a checkfile, passed to stdin.
        let output = cmd!(b3sum_exe(), "--check").stdin_bytes(expected_checkfile.as_bytes()).dir(dir.path()).stdout_capture().stderr_capture().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_check_output = "\
         a: OK\n\
         b: OK\n\
         c/d: OK\n";
        assert_eq!(expected_check_output, stdout);
        assert_eq!("", stderr);

        // Check the same file, but with Windows-style newlines.
        let windows_style = expected_checkfile.replace("\n", "\r\n");
        let output = cmd!(b3sum_exe(), "--check").stdin_bytes(windows_style.as_bytes()).dir(dir.path()).stdout_capture().stderr_capture().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_check_output = "\
         a: OK\n\
         b: OK\n\
         c/d: OK\n";
        assert_eq!(expected_check_output, stdout);
        assert_eq!("", stderr);

        // Now pass the same checkfile twice on the command line just for fun.
        let checkfile_path = dir.path().join("checkfile");
        fs::write(&checkfile_path, &expected_checkfile).unwrap();
        let output = cmd!(b3sum_exe(), "--check", &checkfile_path, &checkfile_path).dir(dir.path()).stdout_capture().stderr_capture().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let mut double_check_output = String::new();
        double_check_output.push_str(&expected_check_output);
        double_check_output.push_str(&expected_check_output);
        assert_eq!(double_check_output, stdout);
        assert_eq!("", stderr);

        // Corrupt one of the files and check again.
        fs::write(dir.path().join("b"), b"CORRUPTION").unwrap();
        let output = cmd!(b3sum_exe(), "--check", &checkfile_path).dir(dir.path()).stdout_capture().stderr_capture().unchecked().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_check_failure = "\
        a: OK\n\
        b: FAILED\n\
        c/d: OK\n";
        assert!(!output.status.success());
        assert_eq!(expected_check_failure, stdout);
        assert_eq!("b3sum: WARNING: 1 computed checksum did NOT match\n", stderr,);

        // Delete one of the files and check again.
        fs::remove_file(dir.path().join("b")).unwrap();
        let open_file_error = fs::File::open(dir.path().join("b")).unwrap_err();
        let output = cmd!(b3sum_exe(), "--check", &checkfile_path).dir(dir.path()).stdout_capture().stderr_capture().unchecked().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_check_failure = format!(
            "a: OK\n\
         b: FAILED ({})\n\
         c/d: OK\n",
            open_file_error,
        );
        assert!(!output.status.success());
        assert_eq!(expected_check_failure, stdout);
        assert_eq!("b3sum: WARNING: 1 computed checksum did NOT match\n", stderr,);

        // Confirm that --quiet suppresses the OKs but not the FAILEDs.
        let output = cmd!(b3sum_exe(), "--check", "--quiet", &checkfile_path).dir(dir.path()).stdout_capture().stderr_capture().unchecked().run().unwrap();
        let stdout = std::str::from_utf8(&output.stdout).unwrap();
        let stderr = std::str::from_utf8(&output.stderr).unwrap();
        let expected_check_failure = format!("b: FAILED ({})\n", open_file_error);
        assert!(!output.status.success());
        assert_eq!(expected_check_failure, stdout);
        assert_eq!("b3sum: WARNING: 1 computed checksum did NOT match\n", stderr,);
    }
}

#[test]
fn test_check_invalid_characters() {
    blake3::initialize();
    // Check that a null character in the path fails.
    let output = cmd!(b3sum_exe(), "--check")
        .stdin_bytes("0000000000000000000000000000000000000000000000000000000000000000  \0")
        .stdout_capture()
        .stderr_capture()
        .unchecked()
        .run()
        .unwrap();
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    let expected_stderr = "\
        b3sum: Null character in path\n\
        b3sum: WARNING: 1 computed checksum did NOT match\n";
    assert!(!output.status.success());
    assert_eq!("", stdout);
    assert_eq!(expected_stderr, stderr);

    // Check that a Unicode replacement character in the path fails.
    let output = cmd!(b3sum_exe(), "--check")
        .stdin_bytes("0000000000000000000000000000000000000000000000000000000000000000  �")
        .stdout_capture()
        .stderr_capture()
        .unchecked()
        .run()
        .unwrap();
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    let expected_stderr = "\
        b3sum: Unicode replacement character in path\n\
        b3sum: WARNING: 1 computed checksum did NOT match\n";
    assert!(!output.status.success());
    assert_eq!("", stdout);
    assert_eq!(expected_stderr, stderr);

    // Check that an invalid escape sequence in the path fails.
    let output = cmd!(b3sum_exe(), "--check")
        .stdin_bytes("\\0000000000000000000000000000000000000000000000000000000000000000  \\a")
        .stdout_capture()
        .stderr_capture()
        .unchecked()
        .run()
        .unwrap();
    let stdout = std::str::from_utf8(&output.stdout).unwrap();
    let stderr = std::str::from_utf8(&output.stderr).unwrap();
    let expected_stderr = "\
        b3sum: Invalid backslash escape\n\
        b3sum: WARNING: 1 computed checksum did NOT match\n";
    assert!(!output.status.success());
    assert_eq!("", stdout);
    assert_eq!(expected_stderr, stderr);
}

#[test]
fn test_globbing() {
    blake3::initialize();
    // Globbing is the shell's.
    let dir = tempfile::tempdir().unwrap();
    let file1 = dir.path().join("file1");
    fs::write(&file1, b"foo").unwrap();
    let file2 = dir.path().join("file2");
    fs::write(&file2, b"bar").unwrap();

    let foo_hash = blake3::convenience::hash(b"foo");
    let bar_hash = blake3::convenience::hash(b"bar");
    // NOTE: This assumes that the glob will be expanded in alphabetical order,
    //       to "file1 file2" rather than "file2 file1". So far, this seems to
    //       be true (guaranteed?) of Unix shell behavior. It's possible that this could
    //       start failing in the future, though, or on some unknown platform.
    //       If that ever happens, we'll need to relax this test somehow,
    //       probably by just testing for both possible outputs. I'm not
    //       handling that case in advance, though, because I'd prefer to hear
    //       about it if it comes up.
    let expected = format!("{}  file1\n{}  file2", foo_hash.to_hex(), bar_hash.to_hex());

    let star_command = format!("{} *", b3sum_exe().to_str().unwrap());
    let output = cmd!("/bin/sh", "-c", star_command).dir(dir.path()).read().unwrap();
    assert_eq!(expected, output);
}

/// --num-threads 1 and 3 hash as the default does: a file past a read
/// piece and the pool's threshold beside a small one, and the same bytes
/// from stdin, every digest the library's. 0 is refused.
#[test]
fn test_num_threads() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    let big: Vec<u8> = (0..9u32 << 20).map(|i| (i * 7 + i / 4093) as u8).collect();
    let (file1, file2) = (dir.path().join("big"), dir.path().join("small"));
    fs::write(&file1, &big).unwrap();
    fs::write(&file2, b"bar").unwrap();
    let expected = format!("{}\n{}", blake3::convenience::hash(&big).to_hex(), blake3::convenience::hash(b"bar").to_hex());
    for threads in ["1", "3"] {
        let output = cmd!(b3sum_exe(), "--num-threads", threads, "--no-names", &file1, &file2).read().unwrap();
        assert_eq!(expected, output, "{threads} threads");
        let output = cmd!(b3sum_exe(), "--num-threads", threads, "--no-names").stdin_bytes(big.clone()).read().unwrap();
        assert_eq!(blake3::convenience::hash(&big).to_hex().to_string(), output, "{threads} threads, stdin");
    }
    let zero = cmd!(b3sum_exe(), "--num-threads", "0").stdin_bytes("foo").stderr_null().unchecked().run().unwrap();
    assert!(!zero.status.success(), "0 threads is refused");
}

/// Inputs around the 4 MiB read piece (one piece, one byte more, two
/// pieces, and more than two), as files and through stdin: each gives
/// hash()'s digest.
#[test]
fn test_inputs_around_the_read_piece() {
    blake3::initialize();
    let dir = tempfile::tempdir().unwrap();
    for len in [(3usize << 20) + 12345, 4 << 20, (4 << 20) + 1, 8 << 20, (9 << 20) + 12345] {
        let path = dir.path().join(format!("input-{len}"));
        let data: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        fs::write(&path, &data).unwrap();
        let expected = blake3::convenience::hash(&data).to_hex().to_string();
        let output = cmd!(b3sum_exe(), &path).read().unwrap();
        assert_eq!(&output[..64], expected, "file of {len} bytes");
        let output = cmd!(b3sum_exe()).stdin_bytes(data).read().unwrap();
        assert_eq!(&output[..64], expected, "stdin of {len} bytes");
    }
}
/// `--num-threads 1` reads and hashes on one thread (no pool), from stdin
/// and from a file read rather than mapped, over several pieces: the same
/// digests as the default, and no file reported changed.
#[test]
fn test_one_thread_matches_the_default() {
    let input: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
    blake3::initialize();
    let expected = blake3::convenience::hash(&input).to_hex().to_string();
    for threads in [None, Some("1")] {
        let args: Vec<&str> = threads.map_or(vec!["--no-names"], |n| vec!["--no-names", "--num-threads", n]);
        let out = cmd(b3sum_exe(), &args).stdin_bytes(input.clone()).read().unwrap();
        assert_eq!(out, expected, "stdin, --num-threads {threads:?}");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("file");
        std::fs::write(&path, &input).unwrap();
        let mut file_args = args.clone();
        file_args.extend(["--no-mmap", path.to_str().unwrap()]);
        let out = cmd(b3sum_exe(), &file_args).read().unwrap();
        assert_eq!(out, expected, "a file read, --num-threads {threads:?}");
    }
}

/// Pseudo-files among many (the batched path on Linux): each hashes as its
/// content, whatever size it reports (/proc's report 0, /sys's 4096).
#[test]
#[cfg(target_os = "linux")]
fn test_pseudo_files_among_many() {
    blake3::initialize();
    let names = ["/proc/version", "/sys/kernel/mm/transparent_hugepage/enabled", "/proc/version"];
    let present: Vec<&str> = names.into_iter().filter(|p| fs::read(p).is_ok()).collect();
    if present.len() < 2 {
        return;
    }
    let output = cmd(b3sum_exe(), std::iter::once("--no-names").chain(present.iter().copied())).stdout_capture().stderr_capture().unchecked().run().unwrap();
    assert!(output.status.success(), "{}", String::from_utf8_lossy(&output.stderr));
    let expected: Vec<String> = present.iter().map(|p| blake3::convenience::hash(&fs::read(p).unwrap()).to_hex().to_string()).collect();
    assert_eq!(expected.join("\n"), String::from_utf8_lossy(&output.stdout).trim_end());
}

/// --sigilcode writes each hash as a sigil, in every output form, and with
/// --check reads sigils and hex side by side. Without it, b3sum writes and
/// reads hex alone, as before.
#[test]
fn test_sigilcode() {
    // BLAKE3 of the empty input, as johnservil/sigil's README gives it.
    const EMPTY: &str = "ᢳᕌᔃጾዿᑂഔጕꔍⶰᣟꕄꕛቴⴲజᐽᙿᕯⶂጁᑃሃᛜጩ";
    let dir = tempfile::tempdir().unwrap();
    let empty = dir.path().join("empty");
    let full = dir.path().join("full");
    fs::write(&empty, b"").unwrap();
    fs::write(&full, vec![7u8; 1 << 20]).unwrap();
    let out = cmd!(b3sum_exe(), "--sigilcode", "--no-names", &empty).read().unwrap();
    assert_eq!(out, EMPTY);
    let tagged = cmd!(b3sum_exe(), "--sigilcode", "--tag", &empty).read().unwrap();
    assert_eq!(tagged, format!("BLAKE3 ({}) = {EMPTY}", empty.to_string_lossy()));
    // A sigil of any length from one byte, its letters three bytes of UTF-8
    // each; no output at all is a clear error.
    for (len, letters) in [("1", 2), ("16", 13), ("64", 48), ("1000", 729)] {
        let sigil = cmd!(b3sum_exe(), "--sigilcode", "--no-names", "-l", len, &full).read().unwrap();
        assert_eq!((sigil.chars().count(), sigil.len()), (letters, 3 * letters), "{len}");
    }
    let f = cmd!(b3sum_exe(), "--sigilcode", "-l", "0", &full).stderr_capture().unchecked().run().unwrap();
    assert!(!f.status.success());
    assert!(String::from_utf8_lossy(&f.stderr).contains("--sigilcode writes at least one output byte"));
    // A checkfile mixing sigils, hex, and the tagged form: --sigilcode
    // checks it clean, and plain --check refuses its sigil lines.
    let sigil_line = cmd!(b3sum_exe(), "--sigilcode", &full).read().unwrap();
    let hex_line = cmd!(b3sum_exe(), &empty).read().unwrap();
    let checkfile = dir.path().join("sums.b3");
    fs::write(&checkfile, format!("{sigil_line}\n{hex_line}\n{tagged}\n")).unwrap();
    let checked = cmd!(b3sum_exe(), "--sigilcode", "--check", &checkfile).read().unwrap();
    assert_eq!(checked.lines().filter(|l| l.ends_with(": OK")).count(), 3, "{checked}");
    let plain = cmd!(b3sum_exe(), "--check", &checkfile).stdout_capture().stderr_capture().unchecked().run().unwrap();
    assert!(!plain.status.success());
    assert_eq!(String::from_utf8_lossy(&plain.stdout).lines().filter(|l| l.ends_with(": OK")).count(), 1);
    assert_eq!(String::from_utf8_lossy(&plain.stderr).matches("a sigil? check it with --sigilcode").count(), 2);
    // Two neighbours swapped: the sigil's own check refuses it.
    let (sigil, name) = sigil_line.split_once("  ").unwrap();
    let mut chars: Vec<char> = sigil.chars().collect();
    chars.swap(3, 4);
    fs::write(&checkfile, format!("{}  {name}\n", chars.iter().collect::<String>())).unwrap();
    let f = cmd!(b3sum_exe(), "--sigilcode", "--check", &checkfile).stderr_capture().unchecked().run().unwrap();
    assert!(!f.status.success());
    assert!(String::from_utf8_lossy(&f.stderr).contains("Invalid hash, read as a sigil"));
    // Raw bytes have no text form.
    let f = cmd!(b3sum_exe(), "--sigilcode", "--raw", &empty).stderr_capture().unchecked().run().unwrap();
    assert!(!f.status.success());
}
