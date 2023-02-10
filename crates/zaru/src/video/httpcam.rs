//! IP webcam client for servers offering HTTP MJPG streams.

use std::{
    io::{self, prelude::*, BufRead, BufReader},
    net::{SocketAddr, TcpStream},
};

use crate::image::Image;
use crate::timer::Timer;
use anyhow::bail;

pub struct HttpStream {
    stream: BufReader<TcpStream>,
    boundary: String,
    t_dequeue: Timer,
    t_decode: Timer,
}

impl HttpStream {
    pub fn connect(addr: SocketAddr) -> anyhow::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        write!(stream, "GET /video HTTP/1.1\r\nHost: {}\r\n\r\n", addr.ip())?;

        let mut stream = BufReader::new(stream);
        let mut line = String::new();
        stream.read_line(&mut line)?;
        log::trace!("response: {}", line.trim());
        if !line.starts_with("HTTP/1.1 200") {
            bail!("received unexpected response: {}", line.trim());
        }

        let mut boundary = None;
        while !line.is_empty() {
            line.clear();
            stream.read_line(&mut line)?;
            log::trace!("response header: {}", line.trim());
            let Some((name, value)) = line.split_once(':') else {
                bail!("malformed HTTP response");
            };
            if name.eq_ignore_ascii_case("Content-Type") {
                let Some((mime, bnd)) = value.trim().split_once(';') else {
                    bail!("malformed Content-Type header");
                };
                if mime != "multipart/x-mixed-replace" {
                    bail!("malformed Content-Type header: unexpected mime type {mime}");
                }
                let Some(bnd) = bnd.strip_prefix("boundary=") else {
                    bail!("malformed Content-Type header (missing boundary)");
                };
                log::trace!("multipart boundary: {bnd}");
                let mut bnd = bnd.to_string();
                // Some servers (Droidcam) include the `--` in the boundary specification. This
                // appears to violate the MIME spec.
                if !bnd.starts_with("--") {
                    bnd = format!("--{bnd}");
                }
                boundary = Some(bnd);
                break;
            }
        }

        let Some(boundary) = boundary else {
            bail!("missing `Content-Type` header");
        };

        let mut this = Self {
            stream,
            boundary,
            t_dequeue: Timer::new("dequeue"),
            t_decode: Timer::new("decode"),
        };
        this.read_until_boundary()?;
        Ok(this)
    }

    pub fn read(&mut self) -> anyhow::Result<Image> {
        let mut length = None;
        let mut line = String::new();
        loop {
            line.clear();
            self.stream.read_line(&mut line)?;
            if line == "\r\n" {
                break;
            }
            log::trace!("multipart header: {}", line.trim());
            let Some((key, value)) = line.split_once(':') else {
                bail!("malformed Content-Type header");
            };
            if key.eq_ignore_ascii_case("Content-Type") {
                if value.trim() != "image/jpeg" {
                    bail!(
                        "unexpected Content-Type: expected image/jpeg, got {}",
                        value.trim()
                    );
                }
            }
            if key.eq_ignore_ascii_case("Content-Length") {
                length = Some(value.trim().parse::<u32>()?);
            }
        }

        let Some(length) = length else {
            bail!("missing Content-Length header");
        };
        let mut buf = vec![0; length as usize];
        self.stream.read_exact(&mut buf)?;
        self.read_until_boundary()?;

        self.t_decode.time(|| Image::decode_jpeg(&buf))
    }

    /// Returns profiling timers for webcam access and decoding.
    pub fn timers(&self) -> impl Iterator<Item = &Timer> + '_ {
        [&self.t_dequeue, &self.t_decode].into_iter()
    }

    fn read_until_boundary(&mut self) -> io::Result<()> {
        let mut line = String::new();
        loop {
            line.clear();
            self.stream.read_line(&mut line)?;
            if line.trim() == self.boundary {
                return Ok(());
            }
        }
    }
}
