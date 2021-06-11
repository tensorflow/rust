//! A module for reading and writing TFRecords, Tensorflow's preferred on-disk data format.
//!
//! See the [tensorflow docs](https://www.tensorflow.org/api_guides/python/python_io#tfrecords-format-details)
//! for details of this format.
use byteorder::{ByteOrder, LittleEndian, ReadBytesExt, WriteBytesExt};
use crc::Crc;
use crc::CRC_32_ISCSI;
use std::{
    error::Error,
    fmt, io,
    io::{Read, Seek, SeekFrom, Write},
};

fn mask_crc(crc: u32) -> u32 {
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282_ead8u32)
}

const CASTAGNOLI: Crc<u32> = Crc::<u32>::new(&CRC_32_ISCSI);

/// A type for writing bytes in the TFRecords format.
#[derive(Debug)]
pub struct RecordWriter<W: Write> {
    writer: W,
}

impl<W> RecordWriter<W>
where
    W: Write,
{
    /// Construct a new RecordWriter which writes to `writer`.
    pub fn new(writer: W) -> Self {
        RecordWriter { writer }
    }

    /// Write a complete TFRecord.
    pub fn write_record(&mut self, bytes: &[u8]) -> io::Result<()> {
        /* A TFRecords file contains a sequence of strings
        with CRC32C (32-bit CRC using the Castagnoli polynomial) hashes. Each record has the format

        uint64 length
        uint32 masked_crc32_of_length
        byte   data[length]
        uint32 masked_crc32_of_data
        and the records are concatenated together to produce the file. CRCs are described here [1],
        and the mask of a CRC is :
        masked_crc = ((crc >> 15) | (crc << 17)) + 0xa282ead8ul

        [1] https://en.wikipedia.org/wiki/Cyclic_redundancy_check
        */
        let mut len_bytes = [0u8; 8];
        (&mut len_bytes[..]).write_u64::<LittleEndian>(bytes.len() as u64)?;

        let masked_len_crc32c = mask_crc(CASTAGNOLI.checksum(&len_bytes));
        let mut len_crc32_bytes = [0u8; 4];
        (&mut len_crc32_bytes[..]).write_u32::<LittleEndian>(masked_len_crc32c)?;

        let masked_bytes_crc32c = mask_crc(CASTAGNOLI.checksum(&bytes));
        let mut bytes_crc32_bytes = [0u8; 4];
        (&mut bytes_crc32_bytes[..]).write_u32::<LittleEndian>(masked_bytes_crc32c)?;

        self.writer.write_all(&len_bytes)?;
        self.writer.write_all(&len_crc32_bytes)?;
        self.writer.write_all(bytes)?;
        self.writer.write_all(&bytes_crc32_bytes)?;
        Ok(())
    }
}

#[derive(Debug)]
/// The possible errors from a record read attempt
pub enum RecordReadError {
    /// This record is corrupt (failed a checksum), but we might be able to recover.
    /// A subsequent call to read_next() might yield the next record.
    CorruptRecord,

    /// The entire file is corrupted. This is a terminal error.
    CorruptFile,

    /// There was an underlying io error. Depending on the source of the error, this may be a
    /// a transient or permanent failure.
    IoError {
        /// The underlying io::Error
        source: io::Error,
    },

    /// The supplied buffer was too short to contain the next record.
    BufferTooShort {
        /// The length of the record that was too long to read.
        needed: u64,
    },
}

impl From<io::Error> for RecordReadError {
    fn from(from: io::Error) -> RecordReadError {
        RecordReadError::IoError { source: from }
    }
}

impl Error for RecordReadError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            RecordReadError::IoError { source } => Some(source),
            _ => None,
        }
    }
}

impl fmt::Display for RecordReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

/// A type for deserializing TFRecord formats
#[derive(Debug)]
pub struct RecordReader<R: Read + Seek> {
    reader: R,
}

impl<R> RecordReader<R>
where
    R: Read + Seek,
{
    /// Construct a new RecordReader from an underlying Read.
    pub fn new(reader: R) -> Self {
        RecordReader { reader }
    }

    fn read_next_len_unchecked(&mut self) -> Result<Option<u64>, RecordReadError> {
        match self.reader.read_u64::<LittleEndian>() {
            Err(e) => {
                if e.kind() == io::ErrorKind::UnexpectedEof {
                    Ok(None)
                } else {
                    Err(e.into())
                }
            }
            Ok(val) => Ok(Some(val)),
        }
    }

    fn checksum(&mut self, bytes: &[u8]) -> Result<bool, RecordReadError> {
        let actual_bytes_crc32 = mask_crc(CASTAGNOLI.checksum(&bytes));
        let expected_bytes_crc32 = self.reader.read_u32::<LittleEndian>()?;
        if actual_bytes_crc32 != expected_bytes_crc32 {
            return Ok(false);
        }
        Ok(true)
    }
    fn read_bytes_exact_unchecked(&mut self, buf: &mut [u8]) -> Result<(), RecordReadError> {
        self.reader.read_exact(buf)?;
        Ok(())
    }

    /// The length of the next record. Does not checksum the length.
    /// Use this to find out how large the byte slice needs to be to read the next record.
    pub fn peek_next_len(&mut self) -> Result<Option<u64>, RecordReadError> {
        let len = self.read_next_len_unchecked()?;
        if len.is_some() {
            self.reader.seek(SeekFrom::Current(-8))?;
        }

        Ok(len)
    }
    /// Read the next record into a byte slice.
    /// Returns the number of bytes read, if successful.
    /// Returns None, if it could read exactly 0 bytes (indicating EOF)
    ///
    /// # Examples
    /// ```
    /// // When we are sure of the max item size, we can just stack allocate an array to hold
    /// use tensorflow::io::{RecordReadError, RecordReader, RecordWriter};
    /// use std::{io::Cursor, rc::Rc};
    /// let mut buf = Vec::new();
    /// let mut rc = Rc::new(&mut buf);
    /// let records = vec!["foo", "barr", "baz"];
    /// {
    ///     let mut writer = RecordWriter::new(Rc::get_mut(&mut rc).unwrap());
    ///     for rec in records.iter() {
    ///         writer.write_record(rec.as_bytes()).unwrap();
    ///     }
    /// }
    /// let read = std::io::BufReader::new(Cursor::new(buf));
    /// let mut reader = RecordReader::new(read);
    /// let mut ary = [0u8; 4];
    /// let mut i = 0;
    /// loop {
    ///     let next = reader.read_next(&mut ary);
    ///     match next {
    ///         Ok(res) => match res {
    ///             Some(len) => assert_eq!(&ary[0..len], records[i].as_bytes()),
    ///             None => break,
    ///         },
    ///         Err(RecordReadError::CorruptFile) | Err(RecordReadError::IoError { .. }) => {
    ///             break;
    ///         }
    ///         _ => {}
    ///     }
    ///     i += 1;
    /// }
    /// ```
    /// When we may need to dynamically resize our buffer, use this peek_next_len()
    /// ```
    /// use tensorflow::io::{RecordReadError, RecordReader, RecordWriter};
    /// use std::{io::Cursor, rc::Rc};
    /// let mut buf = Vec::new();
    /// let mut rc = Rc::new(&mut buf);
    /// let records = vec!["foo", "barr", "baz"];
    /// {
    ///     let mut writer = RecordWriter::new(Rc::get_mut(&mut rc).unwrap());
    ///     for rec in records.iter() {
    ///         writer.write_record(rec.as_bytes()).unwrap();
    ///     }
    /// }
    /// let read = std::io::BufReader::new(Cursor::new(buf));
    /// let mut reader = RecordReader::new(read);
    /// let mut vec = Vec::new();
    /// while let Ok(Some(len)) = reader.peek_next_len() {
    ///     let len = len as usize;
    ///     if vec.len() < len {
    ///         vec.resize(len, 0);
    ///     }
    ///     let next = reader.read_next(&mut vec[0..len]);
    ///     assert_eq!(next.unwrap().unwrap(), len);
    ///     // &vec[0..len] contains the bytes of this record
    /// }
    /// assert_eq!(vec.len(), 4);
    /// ```
    ///
    pub fn read_next(&mut self, buf: &mut [u8]) -> Result<Option<usize>, RecordReadError> {
        let len = match self.read_next_len_unchecked()? {
            Some(len) => len,
            None => return Ok(None),
        };
        if (buf.len() as u64) < len {
            self.reader.seek(SeekFrom::Current(8 + len as i64))?;
            return Err(RecordReadError::BufferTooShort { needed: len });
        }

        let mut len_bytes = [0u8; 8];
        LittleEndian::write_u64(&mut len_bytes, len);
        if !self.checksum(&len_bytes)? {
            return Err(RecordReadError::CorruptFile);
        }

        let slice = &mut buf[0..len as usize];
        self.read_bytes_exact_unchecked(slice)?;

        if self.checksum(slice)? {
            Ok(Some(len as usize))
        } else {
            Err(RecordReadError::CorruptRecord)
        }
    }
    /// Allocate a Vec<u8> on the heap and read the next record into it.
    /// Returns the filled Vec, if successful.
    /// Returns None, if it could read exactly 0 bytes (indicating EOF)
    /// # Example
    /// ```
    /// use tensorflow::io::{RecordReadError, RecordReader, RecordWriter};
    /// use std::{io::Cursor, rc::Rc};
    /// let mut buf = Vec::new();
    /// let mut rc = Rc::new(&mut buf);
    /// let records = vec!["foo", "barr", "baz"];
    /// {
    ///     let mut writer = RecordWriter::new(Rc::get_mut(&mut rc).unwrap());
    ///     for rec in records.iter() {
    ///         writer.write_record(rec.as_bytes()).unwrap();
    ///     }
    /// }
    /// let read = std::io::BufReader::new(Cursor::new(buf));
    /// let mut reader = RecordReader::new(read);
    /// let mut i = 0;
    /// loop {
    ///     let next = reader.read_next_owned();
    ///     match next {
    ///         Ok(res) => match res {
    ///             Some(vec) => assert_eq!(&vec[..], records[i].as_bytes()),
    ///             None => break,
    ///         },
    ///         Err(RecordReadError::CorruptFile) | Err(RecordReadError::IoError { .. }) => {
    ///             break;
    ///         }
    ///         _ => {}
    ///     }
    ///     i += 1;
    /// }
    /// ```
    pub fn read_next_owned(&mut self) -> Result<Option<Vec<u8>>, RecordReadError> {
        let len = match self.read_next_len_unchecked()? {
            Some(len) => len,
            None => return Ok(None),
        };
        let mut len_bytes = [0u8; 8];
        LittleEndian::write_u64(&mut len_bytes, len);
        if !self.checksum(&len_bytes)? {
            return Err(RecordReadError::CorruptFile);
        }
        let mut vec = vec![0u8; len as usize];
        self.read_bytes_exact_unchecked(&mut vec)?;
        if self.checksum(&vec)? {
            Ok(Some(vec))
        } else {
            Err(RecordReadError::CorruptRecord)
        }
    }
    /// Convert the Reader into an Iterator<Item = Result<Vec<u8>, RecordReadError>, which iterates
    /// the whole Read. Stops if it finds whole-file corruption.
    /// # Example
    /// ```
    /// use tensorflow::io::{RecordWriter, RecordReader};
    /// let records = vec!["Foo bar baz", "boom bing bang", "sum soup shennaninganner"];
    /// let path = "test_resources/io/roundtrip.tfrecord";
    /// let out = ::std::fs::OpenOptions::new()
    ///     .write(true)
    ///     .create(true)
    ///     .open(path)
    ///     .unwrap();
    /// {
    ///     let mut writer = RecordWriter::new(out);
    ///     for rec in records.iter() {
    ///         writer.write_record(rec.as_bytes()).unwrap();
    ///     }
    /// }
    /// {
    ///     let actual = ::std::fs::OpenOptions::new().read(true).open(path).unwrap();
    ///     let reader = RecordReader::new(actual);
    ///     for (actual, expected) in reader.into_iter_owned().zip(records) {
    ///         assert_eq!(actual.unwrap(), expected.as_bytes());
    ///     }
    /// }
    /// {
    ///     let actual = ::std::fs::OpenOptions::new().read(true).open(path).unwrap();
    ///     let reader = RecordReader::new(actual);
    ///     assert_eq!(reader.into_iter_owned().count(), 3);
    /// }
    /// let _ = std::fs::remove_file(path);
    /// ```
    pub fn into_iter_owned(self) -> impl Iterator<Item = Result<Vec<u8>, RecordReadError>> {
        RecordOwnedIterator { records: self }
    }
}

struct RecordOwnedIterator<R: Read + Seek> {
    records: RecordReader<R>,
}

impl<R: Read + Seek> Iterator for RecordOwnedIterator<R> {
    type Item = Result<Vec<u8>, RecordReadError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self.records.read_next_owned().transpose() {
            Some(Err(RecordReadError::CorruptFile)) => None,
            rest => rest,
        }
    }
}

////////////////////////

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::File;
    use std::io::Read;

    #[test]
    fn writer_identical_to_python() {
        let actual_filename = "test_resources/io/actual.tfrecord";
        // This file was generated by test_resources/io/python_writer.py
        let expected_filename = "test_resources/io/expected.tfrecord";
        {
            let f = ::std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(actual_filename)
                .unwrap();

            let mut record_writer = RecordWriter::new(::std::io::BufWriter::new(f));
            record_writer
                .write_record("The Quick Brown Fox".as_bytes())
                .unwrap();
        }
        {
            let mut af = File::open(actual_filename).unwrap();
            let mut ef = File::open(expected_filename).unwrap();
            let mut actual = vec![0; 0];
            let mut expected = vec![0; 0];
            af.read_to_end(&mut actual).unwrap();
            ef.read_to_end(&mut expected).unwrap();

            assert_eq!(actual, expected);
        }

        let _ = std::fs::remove_file(actual_filename);
    }
    #[test]
    fn peek_next() {
        use std::{io::Cursor, rc::Rc};
        let mut buf = Vec::new();
        let mut rc = Rc::new(&mut buf);
        let records = vec!["foo", "barr", "baz"];
        {
            let mut writer = RecordWriter::new(Rc::get_mut(&mut rc).unwrap());
            for rec in records.iter() {
                writer.write_record(rec.as_bytes()).unwrap();
            }
        }
        let read = std::io::BufReader::new(Cursor::new(buf));
        let mut reader = RecordReader::new(read);
        let mut ary = [0u8; 4];
        let mut i = 0;
        loop {
            if i < 3 {
                assert_eq!(
                    reader.peek_next_len().unwrap().unwrap() as usize,
                    records[i].len()
                );
            }
            if i == 3 {
                assert!(reader.peek_next_len().unwrap().is_none());
            }

            let next = reader.read_next(&mut ary);
            match next {
                Ok(res) => match res {
                    Some(len) => assert_eq!(&ary[0..len], records[i].as_bytes()),
                    None => break,
                },
                Err(e @ _) => {
                    panic!("Received an unexpected error: {:?}", e);
                }
            }
            i += 1;
        }
    }

    #[test]
    fn seek_over_too_long() {
        use std::{io::Cursor, rc::Rc};
        let mut buf = Vec::new();
        let mut rc = Rc::new(&mut buf);
        let records = vec!["foo", "barr", "baz"];
        {
            let mut writer = RecordWriter::new(Rc::get_mut(&mut rc).unwrap());
            for rec in records.iter() {
                writer.write_record(rec.as_bytes()).unwrap();
            }
        }
        let read = std::io::BufReader::new(Cursor::new(buf));
        let mut reader = RecordReader::new(read);
        let mut ary = [0u8; 3];

        let next = reader.read_next(&mut ary);
        assert_eq!(next.unwrap().unwrap(), 3);
        assert_eq!(&ary, records[0].as_bytes());

        let next = reader.read_next(&mut ary);
        let buffer_too_short = match next {
            Err(RecordReadError::BufferTooShort { needed: v }) => v == 4,
            _ => false,
        };
        assert!(buffer_too_short);

        let next = reader.read_next(&mut ary);
        assert_eq!(next.unwrap().unwrap(), 3);
        assert_eq!(&ary, records[2].as_bytes());

        let next = reader.read_next(&mut ary);
        assert!(next.unwrap().is_none());
    }
}
