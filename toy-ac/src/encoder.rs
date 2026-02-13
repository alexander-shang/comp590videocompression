use super::range::Range;
use super::symbol_model::SymbolModel;
use std::io::Write;
use bitbit::BitWriter;

#[derive(Debug)]
pub struct Encoder {
    range: Range,
    pending: u32,
    finished: bool,
    bits_written: u64
}

impl Encoder {
    pub fn new() -> Self {
        Self {
            range: Range::new(32),
            pending: 0,
            finished: false,
            bits_written: 0
        }
    }

    pub fn encode<T: Eq, W: Write>(&mut self, s: &T, m: &dyn SymbolModel<T>, output: &mut BitWriter<W>) {
        if self.finished {
            panic!("Encoder already finished");
        }
        if !m.contains(s) {
            panic!("Value is not in model");
        }

        let (int_start, int_end) = m.interval(s);        
        let total = m.total() as u64;
        let range_width = self.range.width();
        let low = self.range.low();

        let new_low = low + (range_width * int_start as u64) / total;
        let new_high = low + (range_width * int_end as u64) / total - 1;

        self.range.reduce(new_high, new_low);
        if self.range.hob_match() {
            let is_one = self.range.shift_hob();

            output.write_bit(is_one).unwrap();
            self.bits_written += 1;

            for _ in 0..self.pending {
                output.write_bit(!is_one).unwrap();
                // We don't increment written counter when pending bits 
                // actually written. Already accounted for when pending counter increments.
            }
            self.pending = 0;
            while self.range.hob_match() {
                output.write_bit(self.range.shift_hob()).unwrap();
                self.bits_written += 1;
            }
        }
        assert!(!self.range.hob_match());
        while self.range.in_middle() {
            self.range.shift_sob();
            self.pending += 1;
            self.bits_written += 1;
        }
    }

    pub fn high(&self) -> u64 {
        self.range.high()
    }

    pub fn low(&self) -> u64 {
        self.range.low()
    }

    pub fn finish<W: Write>(&mut self,  output: &mut BitWriter<W>) -> Result<(), Box<dyn std::error::Error>> {
        // Write out any value between range low and high (0x80000000 for example)
        // plus any pending bits as 0. The correct understanding of this is 
        // writing out a 1, plus any pending bits as 0, followed by 31 more zeroes.

        output.write_bit(true)?;
        self.bits_written += 1;
        for _ in 0..self.pending+31 {
            output.write_bit(false)?;
            self.bits_written += 1;
        }

        self.finished = true;
        Ok(())
    }

    pub fn bits_written(&self) -> u64 {
        return self.bits_written;
    }
}

