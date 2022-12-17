//! [Droidcam] IP webcam client.
//!
//! [Droidcam]: https://droidcam.org/

use std::{
    io::{self, prelude::*, BufRead, BufReader},
    net::{SocketAddr, TcpStream},
};

use crate::{image::Image, timer::Timer};

pub struct Droidcam {
    stream: BufReader<TcpStream>,
    boundary: String,
    t_dequeue: Timer,
    t_decode: Timer,
}

impl Droidcam {
    pub const DEFAULT_PORT: u16 = 4747;

    pub fn connect(addr: SocketAddr) -> crate::Result<Self> {
        let mut stream = TcpStream::connect(addr)?;
        write!(stream, "GET /video HTTP/1.1\r\nHost: {}\r\n\r\n", addr.ip())?;

        let mut stream = BufReader::new(stream);
        let mut line = String::new();
        stream.read_line(&mut line)?;
        log::trace!("response: {}", line.trim());
        if !line.starts_with("HTTP/1.1 200") {
            return Err(format!("received unexpected response: {}", line.trim()).into());
        }

        let mut boundary = None;
        while !line.is_empty() {
            line.clear();
            stream.read_line(&mut line)?;
            log::trace!("response header: {}", line.trim());
            let Some((name, value)) = line.split_once(':') else {
                return Err("malformed HTTP response".into());
            };
            if name.eq_ignore_ascii_case("Content-Type") {
                let Some((mime, bnd)) = value.trim().split_once(';') else {
                    return Err("malformed Content-Type header".into());
                };
                if mime != "multipart/x-mixed-replace" {
                    return Err(format!(
                        "malformed Content-Type header: unexpected mime type {mime}"
                    )
                    .into());
                }
                let Some(bnd) = bnd.strip_prefix("boundary=") else {
                    return Err("malformed Content-Type header (missing boundary)".into());
                };
                boundary = Some(bnd.to_string());
                break;
            }
        }

        let Some(boundary) = boundary else {
            return Err("missing `Content-Type` header".into());
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

    pub fn read(&mut self) -> Result<Image, crate::Error> {
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
                return Err("malformed Content-Type header".into());
            };
            if key.eq_ignore_ascii_case("Content-Type") {
                if value.trim() != "image/jpeg" {
                    return Err(format!(
                        "unexpected Content-Type: expected image/jpeg, got {}",
                        value.trim()
                    )
                    .into());
                }
            }
            if key.eq_ignore_ascii_case("Content-Length") {
                length = Some(value.trim().parse::<u32>()?);
            }
        }

        let Some(length) = length else {
            return Err("missing Content-Length header".into());
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
