use std::f64::consts::LOG2_E;

const BYTE: u64 = 8;
#[allow(dead_code)]

pub struct APBFSlice {
    /// Number of bits in the slice. Equal to `length * 8` (or `capacity`) of the `Vec` in the
    /// `data` field, if it's not unexpectedly underfilled and no reallocation has happened.
    pub size: u64,
    /// Number of bits set in the slice.
    set_bits: u64,
    /// The index of the slice's hash function. Should be equal to the hash_index of slices that
    /// are `num_hashes` indices away in the insertion direction (here ->) of the slices' circular
    /// buffer, since only `num_hashes` different hash functions are needed. Are used for hash
    /// reuse during queries and as seeds of the hash function used for insertion and querying.
    pub hash_index: usize,
    /// The array of bytes that hold the slice's membership information. Is should always be filled
    /// to its maximum capacity (`length == capacity`).
    data: Vec<u8>,
    /// For use with probabilistic CRDTs for synchronizations. Holds only the current generation if
    /// the slice is in the interval of the first `k` slices of the circular buffer (i.e., is an
    /// active slice). It is cleared every shift, preparing for the arrival of a new generation.
    pub current_gen: Vec<u8>,
}

impl APBFSlice {
    /// Unions the bits of the elements inserted at slice `other` with those of slice `self`,
    /// as long as they have the same dimensions. In the context of the APBF, both must have
    /// the same hash index, so that the bits set at `other` can be queried at `self`.
    fn union(&mut self, other: &Self) {
        // Slices must have the same size
        assert_eq!(self.size, other.size);
        assert_eq!(self.data.len(), other.data.len());
        assert_eq!(self.data.len(), other.current_gen.len());
        for i in 0..self.data.len() {
            // Choosing exclusive bits to each of the bytes.
            let xor = self.data[i] ^ other.data[i];
            // Choosing exclusive bits set by `other` (sbo).
            let sbo = other.data[i] & xor;
            // Counting the number of bits set by `other`.
            let bc = improved_count_set_bits_u8(sbo);
            // Updating the set bit count of the slice.
            self.set_bits += bc as u64;

            // Merging the contents.
            self.data[i] |= other.data[i];
        }
    }

    /// Unions the bits of the elements of the current generation inserted at slice `other` with
    /// those of slice `self`, as long as they have the same dimensions. In the context of the APBF,
    /// both must have the same hash index, so that the bits set at `other` can be queried at `self`.
    pub fn union_cur_gen(&mut self, other: &Self) {
        // Slices must have the same size
        assert_eq!(self.size, other.size);
        assert_eq!(self.data.len(), other.data.len());
        assert_eq!(self.data.len(), other.current_gen.len());
        for i in 0..self.data.len() {
            // Choosing exclusive bits to each of the bytes.
            let xor = self.data[i] ^ other.current_gen[i];
            // Choosing exclusive bits set by `other` (sbo).
            let sbo = other.current_gen[i] & xor;
            // Counting the number of bits set by `other`.
            let bc = improved_count_set_bits_u8(sbo);
            // Updating the set bit count of the slice.
            self.set_bits += bc as u64;

            // Merging the contents.
            self.data[i] |= other.current_gen[i];
            // // Experimental (for transitive propagations)
            // self.current_gen[i] |= other.current_gen[i];
        }
    }

    /// Returns a 64-bit float in the interval `[0, 1]` that indicates the ratio of set bits to
    /// total bits in the slice.
    pub fn fill_ratio(&self) -> f64 {
        self.set_bits as f64 / self.size as f64
    }

    /// Returns a 64-bit float in the interval `[0, 1]` that indicates the ratio of set bits to
    /// total bits in the slice.
    pub fn counting_fill_ratio(&self) -> f64 {
        self.count_set_data_bits() as f64 / self.size as f64
    }

    /// Counts the set bits of the data vector of the slice.
    pub fn count_set_data_bits(&self) -> u64 {
        count_set_bits(&self.data)
    }

    /// Counts the set bits of the current_gen vector of the slice.
    pub fn count_set_current_gen_bits(&self) -> u64 {
        count_set_bits(&self.current_gen)
    }

    /// Given a location `loc` - an index of a bit in the data vector of bytes - returns a tuple
    /// indicating the index of the byte the bit belongs to and its index within that byte.
    fn get_byte_and_bit_idxs(&self, loc: u64) -> (usize, u8) {
        assert!(loc < 8 * self.data.len() as u64);
        let byte_idx = (loc / BYTE) as usize;
        let bit_idx = 7 - (loc % BYTE) as u8;
        // Alternative endianness
        // let bit_idx = (loc % BYTE) as u8;
        (byte_idx, bit_idx)
    }

    /// Returns `true` if the bit at location `loc` is set to `1`; `false` otherwise.
    fn get_bit_at_loc(&self, loc: u64) -> bool {
        let (byte_idx, bit_idx) = self.get_byte_and_bit_idxs(loc);
        // Is 0000 0000 when false; has one of the bits set to 1 when true
        let bit_set = self.data[byte_idx] & 1 << bit_idx;
        bit_set != 0
    }

    /// Returns `true` if the bit at byte index `byte_idx` and bit index `bit_idx` within it is
    /// set to `1`; `false` otherwise.
    fn get_bit_at_idxs(&self, byte_idx: usize, bit_idx: u8) -> bool {
        // Is 0000 0000 when false; has one of the bits set to 1 when true
        let bit_set = self.data[byte_idx] & 1 << bit_idx;
        bit_set != 0
    }

    /// Sets the value of the bit at location `loc` to `1`.
    #[allow(dead_code)]
    fn set_bit_at_loc(&mut self, loc: u64) {
        let (byte_idx, bit_idx) = self.get_byte_and_bit_idxs(loc);
        self.data[byte_idx] |= 1 << bit_idx;
        self.current_gen[byte_idx] |= 1 << bit_idx;
    }

    /// Sets the bit at byte index `byte_idx` and bit index `bit_idx` within it to `1`.
    fn set_bit_at_idxs(&mut self, byte_idx: usize, bit_idx: u8) {
        self.data[byte_idx] |= 1 << bit_idx;
        self.current_gen[byte_idx] |= 1 << bit_idx;
    }

    /// Returns `true` if the bit at location `loc` was previously set to `1`; `false` otherwise,
    /// now setting it to `1`.
    #[allow(dead_code)]
    fn get_set_bit_at_loc(&mut self, loc: u64) -> bool {
        let (byte_idx, bit_idx) = self.get_byte_and_bit_idxs(loc);
        let previously_set = self.get_bit_at_idxs(byte_idx, bit_idx);
        if !previously_set {
            self.set_bit_at_idxs(byte_idx, bit_idx);
        }
        previously_set
    }

    /// Creates a slice of an APBF, given the number of bits its data vector should have at least.
    /// This is given by parameter `size`, to be aligned to 64-bit blocks. One other parameter is
    /// `hash_index`, used by the APBF as a seed for hashing and, consequently, for deciding which
    /// bit of the slice is of interest for queries and insertions.
    pub fn create(size: u64, hash_index: usize) -> APBFSlice {
        let size_ceil64 = ceil64(size);
        APBFSlice {
            size: size_ceil64 * 64,
            set_bits: 0,
            hash_index,
            data: vec![0; (size_ceil64 * BYTE) as usize],
            current_gen: vec![0; (size_ceil64 * BYTE) as usize],
        }
    }
}

/// Age-Partitioned Bloom Filter
pub struct APBF {
    /// Indicates the physical index of the first slice of the circular buffer, of logical index 0.
    /// It is an index of the `slices` field, implemented for this struct as a `Vec`.
    /// Starts with value `0`.
    pub base_slice_idx: usize,

    // Low-level variables
    /// k       - number of hash functions used
    pub num_hashers: usize,
    /// l       - number of old batches to store
    pub num_batches: usize,
    /// k + l   - number of slices in the APBF
    pub num_slices: usize,

    // High-level variables
    /// Required error rate
    _error_rate: f64,
    /// Capacity required by the user
    _capacity: u64,
    /// Number of items already inserted
    inserted: u64,
    /// Defines the union mode. If `true`, 2 APBFs will join every one of their slices; if `false`,
    /// they only join their active (first k) slices, according to the `current_gen_union` field.
    pub whole_filter_union: bool,
    /// If union_whole_filter is false, then the union mode will perform the union only over the
    /// active (first k) slices. This field specifies if the union of those active slices will use
    /// the complete slice data or just the most recent generation of elements inserted.
    pub current_gen_union: bool,

    /// Slices of the Age-Partitioned Bloom Filter
    pub slices: Vec<APBFSlice>,
    /// Maximum fill ratios expected for the k slices of the filter
    pub fill_ratios: Vec<f64>,
}

impl Default for APBF {
    fn default() -> Self {
        APBF::create_high_level(2, 4096, 5, false, true)
    }
}

impl APBF {
    /// Given a logical index of a slice of the APBF, returns its respective physical index.
    pub fn log_to_phy_idx(&self, index: usize) -> usize {
        assert!(index < self.num_slices);
        (self.base_slice_idx + index).rem_euclid(self.num_slices)
    }

    /// Given a physical index of a slice of the APBF, returns its respective logical index.
    pub fn phy_to_log_idx(&self, index: usize) -> usize {
        assert!(index < self.num_slices);
        if index < self.base_slice_idx {
            self.num_slices - self.base_slice_idx + index
        } else {
            index - self.base_slice_idx
        }
    }

    /// Unions two APBFs, as long as they have the same mode set for unioning; otherwise panics,
    /// as assertions fail.
    /// Returns the number of shifts executed by an underlying function that aims to preserve the
    /// fill ratio of the active slices.
    pub fn union(&mut self, other: &Self) -> usize {
        // Verifying that the slices being unioned are set to the same unioning mode.
        assert_eq!(self.whole_filter_union, other.whole_filter_union);
        assert_eq!(self.current_gen_union, other.current_gen_union);

        if self.whole_filter_union {
            self.union_whole_filter(other)
        } else {
            self.union_active_slices(other, self.current_gen_union)
        }
    }

    /// Unions the active (first k) slices of two APBFs. Each slice is unioned with its match from
    /// the 'other' filter (same hash index).
    /// If argument `only_cur_gen` is `true`, only the contents of the current generation of
    /// elements inserted at `other` are inserted in the slices at `self`.
    /// Returns the number of shifts executed by an underlying function that aims to preserve the
    /// fill ratio of the active slices.
    fn union_active_slices(&mut self, other: &Self, only_cur_gen: bool) -> usize {
        // The APBFs involved in the union must have the same size (k, l and thus, number of slices)
        assert_eq!(self.num_hashers, other.num_hashers);
        assert_eq!(self.num_batches, other.num_batches);
        assert_eq!(self.num_slices, other.num_slices);
        // Indices of the base slices of 'self' and 'other'
        let self_base_idx = self.base_slice_idx;
        let other_base_idx = other.base_slice_idx;
        // Offset to the base slice of 'other' with the same hash index as the base slice of 'self'
        let mut offset = self.slices[self_base_idx].hash_index as i32
            - other.slices[other_base_idx].hash_index as i32;

        // Misc. variables
        let mut slices_unioned = 0_usize;
        let num_hashers = self.num_hashers;
        let num_slices = self.num_slices;

        // Attempt to union the first k slices with matching hash index and size.
        while slices_unioned < num_hashers {
            offset = offset.rem_euclid(num_hashers as i32);
            let self_slice_idx = (self_base_idx + slices_unioned).rem_euclid(num_slices);
            let matching_slice_idx = (other_base_idx + offset as usize).rem_euclid(num_slices);
            if only_cur_gen {
                self.slices[self_slice_idx].union_cur_gen(&other.slices[matching_slice_idx]);
            } else {
                self.slices[self_slice_idx].union(&other.slices[matching_slice_idx]);
            }
            slices_unioned += 1;
            offset += 1;
        }

        self.check_and_shift()
    }

    /// Unions the filters' matching slices (by physical index). Returns the number of shifts
    /// executed by an underlying function that aims to preserve the fill ratio of the active slices.
    /// Expected to cause mayhem.
    fn union_whole_filter(&mut self, other: &Self) -> usize {
        // The APBFs involved in the union must have the same size (k, l and thus, number of slices).
        assert_eq!(self.num_hashers, other.num_hashers);
        assert_eq!(self.num_batches, other.num_batches);
        assert_eq!(self.num_slices, other.num_slices);

        for i in 0..self.num_slices {
            // Unions slices of `self` with slices of `other` with no regard for their position in
            // the circular buffer. Their possibly very different fill ratios can cause the needed
            // amount of shifting to be high, making the filter forget several generations of
            // elements.
            // The seed of the insertion/query hash functions has to be the physical index.
            self.slices[i].union(&other.slices[i]);
        }

        self.check_and_shift()
    }

    /// Checks current fill ratios and shifts slices a calculated amount of times, so that each
    /// slice in the first k does not exceed its currently desired fill ratio.
    /// Returns the number of shifts executed.
    pub fn check_and_shift(&mut self) -> usize {
        // Calculation of the number of shifts needed to preserve fill ratios of specific slices
        // (active, the first k) under the desired maximum values.
        // let shifts = self.calc_needed_shifts();
        let shifts = self.alt_calc_needed_shifts();

        // Execution of the calculated number of necessary shifts
        self.shift_slices_times(shifts);
        // Alternative
        // for _ in 0..shifts {
        //     self.shift_slices();
        // }

        shifts
    }

    // Returns the number of shifts needed to preserve the invariant fill ratios in the APBF.
    // The fill ratio invariants aplly to the first k slices of the circular buffer, used for
    // insertion. They can be considered the active slices.
    pub fn calc_needed_shifts(&self) -> usize {
        let k = self.num_hashers;
        let mut shifts: usize = 0;

        for log_idx in 0..k {
            let phy_idx = self.log_to_phy_idx(log_idx);
            let fill_ratio = self.slices[phy_idx].fill_ratio();

            let mut cursor = log_idx;
            while cursor < self.num_hashers && fill_ratio >= self.fill_ratios[cursor] {
                cursor += 1;
            }
            if cursor != log_idx {
                shifts = shifts.max(cursor - log_idx);
            }
        }

        shifts
    }

    pub fn alt_calc_needed_shifts(&self) -> usize {
        let k = self.num_hashers;
        let mut shifts: usize = 0;

        for log_idx in 0..k {
            let phy_idx = self.log_to_phy_idx(log_idx);
            let set_bits = self.slices[phy_idx].set_bits;

            let mut cursor = log_idx;
            while cursor < self.num_hashers
                && set_bits > self.current_capacity_of_slice(cursor).ceil() as u64
            {
                cursor += 1;
            }
            if cursor != log_idx {
                shifts = shifts.max(cursor - log_idx);
            }
        }

        shifts
    }

    /// Shifts slices logically to enable the insertion of a new generation of elements. In this
    /// process, the last slice in the circular buffer becomes the first and its contents are cleared.
    /// "The logical shift is then performed by zeroing slice s_{k+l-1}, to be reused as the new
    /// s_{0}, and decrementing the base index, modulo k + l." - page 10
    pub fn shift_slices(&mut self) {
        // Clearance of the current_gen data for the first k slices.
        for offset in 0..self.num_hashers {
            // The function call, even if inlined, adds the assert!() related instructions.
            // The assert  could be removed, as a bound check on access would cause a panic in case
            // of an out of bounds read.
            let idx = self.log_to_phy_idx(offset);
            let mut slice = &mut self.slices[idx];
            slice.current_gen = vec![0; (slice.size / 8) as usize];
        }

        let new_base_slice_idx =
            (self.base_slice_idx as isize - 1).rem_euclid(self.num_slices as isize) as usize;
        // let _new_base_slice_idx = ((self.base_slice_idx as isize - 1 + self.num_slices as isize)
        //     % self.num_slices as isize) as usize;
        let new_hash_index = (self.slices[self.base_slice_idx].hash_index as isize - 1)
            .rem_euclid(self.num_hashers as isize) as usize;
        // let _new_hash_index = ((self.slices[self.base_slice_idx].hash_index as isize - 1
        //     + self.num_hashers as isize)
        //     % self.num_hashers as isize) as usize;

        // Update of the filter's base slice index to the (physical) index of the last (previous) slice
        // in the circular buffer.
        self.base_slice_idx = new_base_slice_idx;

        let new_base_slice = &mut self.slices[new_base_slice_idx];
        let slice_byte_length = new_base_slice.data.capacity();

        // Clearance of the set bits' counter.
        new_base_slice.set_bits = 0;
        // Update of the new base slice's hash index.
        new_base_slice.hash_index = new_hash_index;
        // Creation of a new Vec of the adequate length and initialization with zeroes to substitute
        // the data of the last slice in the circular buffer.
        // This is as if the data was cleared off the slice, so that it can be used for insertions
        // for the next generations.
        new_base_slice.data = vec![0_u8; slice_byte_length];
        // Not allocating a new vector for clearing would be better.
        // Cannot use this feature for now using the stable channel compiler.
        // #![feature(slice_fill)]
        // new_base_slice.data.fill(0);
    }

    /// Shifts the slices of an APBF a given number of `times`, clearing the necessary fields of the
    /// reused slices.
    pub fn shift_slices_times(&mut self, times: usize) {
        if times > 0 {
            // Clearance of the current_gen data for the first k slices of `times` generations.
            for offset in 0..self.num_hashers + times - 1 {
                let idx = self.log_to_phy_idx(offset);
                let mut slice = &mut self.slices[idx];
                slice.current_gen = vec![0; (slice.size / 8) as usize];
            }

            let new_base_slice_idx = (self.base_slice_idx as isize - times as isize)
                .rem_euclid(self.num_slices as isize) as usize;
            let new_hash_index = (self.slices[self.base_slice_idx].hash_index as isize
                - times as isize)
                .rem_euclid(self.num_hashers as isize) as usize;

            // Update of the filter's base slice index to the (physical) index of the last (previous)
            // slice in the circular buffer.
            self.base_slice_idx = new_base_slice_idx;

            let slice_byte_length = self.slices[new_base_slice_idx].data.capacity();

            // Clearance of fields of the base slices of `times` generations.
            for offset in 0..times {
                let idx = self.log_to_phy_idx(offset);
                let mut slice = &mut self.slices[idx];
                // Clearance of the set bits' counter.
                slice.set_bits = 0;
                // Update of the new slices' hash index.
                slice.hash_index = (new_hash_index + offset).rem_euclid(self.num_hashers);
                // Creation of a new Vec of the adequate length and initialization with zeroes to
                // substitute the data of the last slice in the circular buffer.
                // This is as if the data was cleared off the slice, so that it can be used for
                // insertions for the next generations.
                slice.data = vec![0_u8; slice_byte_length];
            }
        }
    }

    // The shift triggering condition will ensure that no active slice (the first k slices) can
    // exceed its maximum capacity.
    // It is possible to calculate how many updates n<sub>i</sub> they can currently store, from
    // their position i ∈ [0, k-1] and size m<sub>i</sub>.

    /// Calculates the current capacity for a slice (how many items it should be able to store up
    /// to the current generation), given its position in the circular buffer.
    /// The argument `index` is a logical index. It should point to one of the slices between
    /// indices 0 and (k - 1) - limits included - in the circular buffer.
    pub fn current_capacity_of_slice(&self, index: usize) -> f64 {
        assert!(index < self.num_hashers);
        let slice = &self.slices[self.log_to_phy_idx(index)];
        (1_f64 - ((index + 1) as f64 / (2 * self.num_hashers) as f64)).ln()
            / (1_f64 - (1_f64 / slice.size as f64)).ln()
    }

    /// APBF builder function that takes low-level parameters.
    pub fn create_low_level(
        num_hashers: usize,
        timeframes: usize,
        slice_size: u64,
        whole_filter_union: bool,
        current_gen_union: bool,
    ) -> APBF {
        let num_slices = num_hashers + timeframes;
        let mut apbf = APBF {
            base_slice_idx: 0,
            num_hashers,
            num_batches: timeframes,
            num_slices,
            _error_rate: 0_f64,
            _capacity: 0,
            inserted: 0,
            whole_filter_union,
            current_gen_union,
            slices: Vec::with_capacity(num_slices),
            fill_ratios: Vec::with_capacity(num_hashers),
        };

        for s in 0..num_slices {
            let apbf_slice = APBFSlice::create(slice_size, s % num_hashers);
            apbf.slices.push(apbf_slice);
        }
        for s in 0..num_hashers {
            // Order of the slice in the sequence
            let ord = (s + 1) as f64;
            apbf.fill_ratios.push(ord / (2.0 * num_hashers as f64));
        }

        apbf
    }

    /// APBF builder function that takes high-level parameters.
    pub fn create_high_level(
        error: u8,
        capacity: usize,
        level: u8,
        whole_filter_union: bool,
        current_gen_union: bool,
    ) -> APBF {
        assert!(
            (1..=5).contains(&error),
            "error = {} (1 <= error: u8 <= 5)",
            error
        );
        if error == 1 {
            assert!(level <= 4, "level = {} (0 <= level: u8 <= 4)", error);
        } else {
            assert!(level <= 5, "level = {} (0 <= level: u8 <= 5)", error);
        }

        let mut k: usize = 0;
        let mut l: usize = 0;

        match error {
            1 => match level {
                0 => {
                    k = 4;
                    l = 3;
                }
                1 => {
                    k = 5;
                    l = 7;
                }
                2 => {
                    k = 6;
                    l = 14;
                }
                3 => {
                    k = 7;
                    l = 28;
                }
                4 => {
                    k = 8;
                    l = 56;
                }
                _ => (),
            },
            2 => match level {
                0 => {
                    k = 7;
                    l = 5;
                }
                1 => {
                    k = 8;
                    l = 8;
                }
                2 => {
                    k = 9;
                    l = 14;
                }
                3 => {
                    k = 10;
                    l = 25;
                }
                4 => {
                    k = 11;
                    l = 46;
                }
                5 => {
                    k = 12;
                    l = 88;
                }
                _ => (),
            },
            3 => match level {
                0 => {
                    k = 10;
                    l = 7;
                }
                1 => {
                    k = 11;
                    l = 9;
                }
                2 => {
                    k = 12;
                    l = 14;
                }
                3 => {
                    k = 13;
                    l = 23;
                }
                4 => {
                    k = 14;
                    l = 40;
                }
                5 => {
                    k = 15;
                    l = 74;
                }
                _ => (),
            },
            4 => match level {
                0 => {
                    k = 14;
                    l = 11;
                }
                1 => {
                    k = 15;
                    l = 15;
                }
                2 => {
                    k = 16;
                    l = 22;
                }
                3 => {
                    k = 17;
                    l = 36;
                }
                4 => {
                    k = 18;
                    l = 63;
                }
                5 => {
                    k = 19;
                    l = 117;
                }
                _ => (),
            },
            5 => match level {
                0 => {
                    k = 17;
                    l = 13;
                }
                1 => {
                    k = 18;
                    l = 16;
                }
                2 => {
                    k = 19;
                    l = 22;
                }
                3 => {
                    k = 20;
                    l = 33;
                }
                4 => {
                    k = 21;
                    l = 54;
                }
                5 => {
                    k = 22;
                    l = 95;
                }
                _ => (),
            },
            _ => (),
        };

        // (?) To avoid a mod 0 on the shift code (May apply to the original C version only.)
        assert!(capacity > l);
        assert!(k != 0 && l != 0, "k = {}, l = {} (0 <= k, l: usize)", k, l);

        let slice_size = (k * capacity) as f64 / (l as f64 * 2_f64.ln());
        APBF::create_low_level(
            k,
            l,
            slice_size.ceil() as u64,
            whole_filter_union,
            current_gen_union,
        )
    }

    /// Inserts an item into the APBF. Returns `true` if the input item is guaranteed to be newly
    /// seen in the first k slices; otherwise, if it is found to be either a duplicate or the
    /// trigger of a false positive prior to the insertion, `false` is returned.
    fn insert(&mut self, item: &[u8]) -> bool {
        // Indicates if the item being inserted is reported as already being present in the
        // filter's first k slices (due to being a duplicate or a trigger of a false positive
        // prior to the insertion).
        let mut dup_or_prev_fp = true;
        // Insertion cursor
        let mut phy_cursor = self.base_slice_idx;
        // Number of insertions left
        let mut insertions_left = self.num_hashers;
        // Pair of hashes used when double hashing is prefered; ignored otherwise.
        let (fst, snd) = if cfg!(feature = "double_hashing") {
            (t1ha::t1ha0(item, 0), t1ha::t1ha0(item, 1))
        } else {
            (0, 0)
        };

        while insertions_left > 0 {
            let slice = &mut self.slices[phy_cursor];
            let seed = if !self.whole_filter_union {
                slice.hash_index as u64
            } else {
                phy_cursor as u64
            };
            let hash = if cfg!(feature = "double_hashing") {
                fst.wrapping_add(seed.wrapping_mul(snd))
            } else {
                t1ha::t1ha0(item, seed)
            };

            let (byte_idx, bit_idx) = slice.get_byte_and_bit_idxs(hash % slice.size);
            let previously_set = slice.get_bit_at_idxs(byte_idx, bit_idx);
            if !previously_set {
                slice.set_bit_at_idxs(byte_idx, bit_idx);
                slice.set_bits += 1;
                // If the bit was not set, the item being inserted cannot be a duplicate or
                // previously trigger of a false positive in the active slices.
                if dup_or_prev_fp {
                    dup_or_prev_fp = false;
                }
            }

            phy_cursor = (phy_cursor + 1).rem_euclid(self.num_slices);
            insertions_left -= 1;
        }

        let new_in_active_slices = !dup_or_prev_fp;
        if cfg!(feature = "count_if_new_in_cur_gen") {
            if new_in_active_slices {
                self.inserted += 1;
            }
        } else {
            self.inserted += 1;
        }

        new_in_active_slices
    }

    /// Inserts an item into the APBF, while checking if there should be a change in generation,
    /// which would shift its slices.
    pub fn checked_insert(&mut self, item: &[u8]) -> (bool, usize) {
        let new_in_active_slices = self.insert(item);
        let shifts = self.check_and_shift();

        (new_in_active_slices, shifts)
    }

    /// Inserts multiple items into the APBF..
    pub fn multiple_insert(&mut self, items: &[&[u8]]) -> Vec<(bool, usize)> {
        items.iter().map(|i| self.checked_insert(i)).collect()
    }

    /// The query algorithm described in the paper that optimizes number of accesses. Queries the
    /// existence of an item in the APBF.
    pub fn query(&self, item: &[u8]) -> bool {
        let mut hashes: Vec<u64> = vec![0; self.num_hashers];
        // Variable c in the paper
        let mut done: usize = 0;
        // Variable p in the paper
        let mut carry: usize = 0;
        // Variable i in the paper (starts at l). Here it is a cursor in the circular buffer and as
        // such, needs to be converted to index the intended slice in the physically ordered vector.
        let mut log_cursor = self.num_batches as isize;
        // Variables only used for the double hashing scenario
        let (fst, snd) = if cfg!(feature = "double_hashing") {
            (t1ha::t1ha0(item, 0), t1ha::t1ha0(item, 1))
        } else {
            (0, 0)
        };

        while log_cursor >= 0 {
            let phy_idx = self.log_to_phy_idx(log_cursor as usize);
            let slice = &self.slices[phy_idx];
            let hash_index = slice.hash_index;
            let reuse_hash = !self.whole_filter_union;
            let hash: u64;
            // Get the hash value for the respective slice, if previously calculated.
            if reuse_hash && hashes[hash_index] != 0 {
                hash = hashes[hash_index];
                // Otherwise, calculate it and save it in the `hashes` array.
            } else {
                let seed = if reuse_hash {
                    hash_index as u64
                } else {
                    phy_idx as u64
                };
                hash = if cfg!(feature = "double_hashing") {
                    fst.wrapping_add(seed.wrapping_mul(snd))
                } else {
                    t1ha::t1ha0(item, seed)
                };
                if reuse_hash {
                    hashes[hash_index] = hash;
                }
            }

            // Search for a sequence of k consecutive slices where the item is reported as present.
            if slice.get_bit_at_loc(hash % slice.size) {
                done += 1;
                if carry + done == self.num_hashers {
                    return true;
                }
                log_cursor += 1;
            } else {
                log_cursor -= self.num_hashers as isize;
                carry = done;
                done = 0;
            }
        }

        false
    }

    /// Queries the existence of multiple items in the APBF.
    pub fn multiple_query(&self, items: &[&[u8]]) -> Vec<bool> {
        items.iter().map(|i| self.query(i)).collect()
    }

    /// Returns the probability that an item not in the filter will be reported as present.
    /// This event is known as a false positive and a call to this function will evaluate its
    /// probability at the instant this method is called.
    pub fn expected_fp_probabilty(&self) -> f64 {
        let mut set_bits: u64 = 0;
        let total_bits = self.slices[0].size * self.num_slices as u64;
        for slice in self.slices.iter() {
            set_bits += slice.set_bits;
        }
        (set_bits as f64 / total_bits as f64).powi(self.num_hashers as i32)
    }

    /// Returns the size of the data of an APBF in bytes.
    pub fn data_size(&self) -> usize {
        // The slice of the first physical slice can be used, since slices have the same size.
        self.slices[0].data.len() * self.num_slices
        // // Assuming slices can have different sizes ...
        // let mut total: usize = 0;
        // for s in self.slices.iter() {
        //     total += s.data.len();
        // }
        // total
    }

    /// Returns the size of the current generation ghost slices of an APBF in bytes.
    pub fn cur_gen_size(&self) -> usize {
        // The slice of the first active slice can be used, since slices have the same size.
        self.slices[self.log_to_phy_idx(0)].current_gen.len() * self.num_hashers
        // // Assuming slices can have different sizes ...
        // let mut total: usize = 0;
        // for i in 0..self.num_hashers {
        //     total += self.slices[self.log_to_phy_idx(i)].current_gen.len();
        // }
        // total
    }

    /// Returns the total size of an APBF in bytes.
    /// This is not the exact number of bytes for the APBF, but a minimum that excludes metadata,
    /// because the current gen is only needed for the active slices.
    pub fn total_size(&self) -> usize {
        self.data_size() + self.cur_gen_size()
    }
}

// # Auxiliary Functions

// ## Miscellaneous

fn ceil64(n: u64) -> u64 {
    let res = n as f64 / 64_f64;
    res.ceil() as u64
}

pub fn calc_bits_per_item(error: f64) -> f64 {
    assert!(error > 0_f64 && error < 1_f64);
    -error.ln() / 2_f64.ln().powi(2)
    // f64::ln(-error) / f64::powi(f64::ln(2.0), 2)
}

pub fn calc_num_slices(precision: u32) -> f64 {
    let fp_prob: f64 = 10_f64.powi(-(precision as i32));
    (1_f64 / fp_prob).log2().ceil()
}

pub fn calc_slice_size(inserted: u64) -> f64 {
    (inserted as f64) / 2_f64.ln()
}

pub fn alt_calc_slice_size(inserted: u64) -> f64 {
    // 1.442695 * (inserted as f64)
    LOG2_E * (inserted as f64)
}

// ## Bit counting functions

pub fn count_set_bits(byte_slice: &[u8]) -> u64 {
    let mut i = 0;
    let mut count: u64 = 0;
    let byte_count = byte_slice.len();

    while i < byte_count {
        let raw_bytes = [
            *byte_slice.get(i).unwrap_or(&0_u8),
            *byte_slice.get(i + 1).unwrap_or(&0_u8),
            *byte_slice.get(i + 2).unwrap_or(&0_u8),
            *byte_slice.get(i + 3).unwrap_or(&0_u8),
            *byte_slice.get(i + 4).unwrap_or(&0_u8),
            *byte_slice.get(i + 5).unwrap_or(&0_u8),
            *byte_slice.get(i + 6).unwrap_or(&0_u8),
            *byte_slice.get(i + 7).unwrap_or(&0_u8),
            *byte_slice.get(i + 8).unwrap_or(&0_u8),
            *byte_slice.get(i + 9).unwrap_or(&0_u8),
            *byte_slice.get(i + 10).unwrap_or(&0_u8),
            *byte_slice.get(i + 11).unwrap_or(&0_u8),
            *byte_slice.get(i + 12).unwrap_or(&0_u8),
            *byte_slice.get(i + 13).unwrap_or(&0_u8),
            *byte_slice.get(i + 14).unwrap_or(&0_u8),
            *byte_slice.get(i + 15).unwrap_or(&0_u8),
        ];
        count += optmized_count_set_bits_u128(u128::from_be_bytes(raw_bytes)) as u64;
        i += 16;
    }

    // Alternative method
    // for byte in self.data.iter() {
    //     count += byte_parallel_count_set_bits(*byte) as u64;
    // }

    count
}

// ### Variations of the Gillies-Miller method for sideways addition

// #### Simple and unoptimized

pub fn count_set_bits_u8(mut bit_block: u8) -> u8 {
    let s = [1, 2, 4];
    let b = [0x55, 0x33, 0x0F];
    for i in 0..8_f64.log(2_f64).ceil() as usize {
        bit_block = ((bit_block >> s[i]) & b[i]) + (bit_block & b[i]);
    }
    bit_block
}

pub fn count_set_bits_u32(mut bit_block: u32) -> u8 {
    let s = [1, 2, 4, 8, 16];
    let b = [
        0x5555_5555,
        0x3333_3333,
        0x0F0F_0F0F,
        0x00FF_00FF,
        0x0000_FFFF,
    ];
    for i in 0..32_f64.log(2_f64).ceil() as usize {
        bit_block = ((bit_block >> s[i]) & b[i]) + (bit_block & b[i]);
    }
    bit_block as u8
}

pub fn count_set_bits_u128(mut bit_block: u128) -> u16 {
    let s = [1, 2, 4, 8, 16, 32, 64];
    let b = [
        0x5555_5555_5555_5555_5555_5555_5555_5555,
        0x3333_3333_3333_3333_3333_3333_3333_3333,
        0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F,
        0x00FF_00FF_00FF_00FF_00FF_00FF_00FF_00FF,
        0x0000_FFFF_0000_FFFF_0000_FFFF_0000_FFFF,
        0x0000_0000_FFFF_FFFF_0000_0000_FFFF_FFFF,
        0x0000_0000_0000_0000_FFFF_FFFF_FFFF_FFFF,
    ];
    for i in 0..128_f64.log(2_f64).ceil() as usize {
        bit_block = ((bit_block >> s[i]) & b[i]) + (bit_block & b[i]);
    }
    bit_block as u16
}

// #### Improved optimization (loop-unrolled and others)

pub fn improved_count_set_bits_u8(bit_block: u8) -> u8 {
    let s = [1, 2, 4];
    let b = [0x55, 0x33, 0x0F];
    let mut bit_count = bit_block - ((bit_block >> 1) & b[0]);
    bit_count = ((bit_count >> s[1]) & b[1]) + (bit_count & b[1]);
    bit_count = ((bit_count >> s[2]) + bit_count) & b[2];

    bit_count
}

pub fn improved_count_set_bits_u32(bit_block: u32) -> u32 {
    let s = [1, 2, 4, 8, 16];
    let b = [
        0x5555_5555,
        0x3333_3333,
        0x0F0F_0F0F,
        0x00FF_00FF,
        0x0000_FFFF,
    ];
    let mut bit_count = bit_block - ((bit_block >> 1) & b[0]);
    bit_count = ((bit_count >> s[1]) & b[1]) + (bit_count & b[1]);
    bit_count = ((bit_count >> s[2]) + bit_count) & b[2];
    bit_count = ((bit_count >> s[3]) + bit_count) & b[3];
    bit_count = ((bit_count >> s[4]) + bit_count) & b[4];

    bit_count
}

// #### Most optimized

/// Counts number of set bits in a 4-byte block. Optimal method for u32, with only 12 operations.
pub fn optmized_count_set_bits_u32(mut bit_block: u32) -> u32 {
    bit_block = bit_block - ((bit_block >> 1) & 0x5555_5555);
    bit_block = (bit_block & 0x3333_3333) + ((bit_block >> 2) & 0x3333_3333);
    (((bit_block + (bit_block >> 4)) & 0x0F0F_0F0F).wrapping_mul(0x0101_0101)) >> 24
}

/// Counts number of set bits in a 16-byte block. Optimal method for u128, with only 12 operations.
// Notes
// - Function `wrapping_mul` might also slow this implementation. If the * operator matches this
// behavior with the `--release` optimized binary, it should not be needed.
// - Could return u8, as the max will obviously be 128 set bits, but I opt to save the conversion
// effort.
pub fn optmized_count_set_bits_u128(mut bit_block: u128) -> u128 {
    bit_block = bit_block - ((bit_block >> 1) & 0x5555_5555_5555_5555_5555_5555_5555_5555);
    bit_block = (bit_block & 0x3333_3333_3333_3333_3333_3333_3333_3333)
        + ((bit_block >> 2) & 0x3333_3333_3333_3333_3333_3333_3333_3333);
    (((bit_block + (bit_block >> 4)) & 0x0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F_0F0F)
        .wrapping_mul(0x0101_0101_0101_0101_0101_0101_0101_0101))
        >> 120
}

// ### Variations of the Brian Kernighan's way

// It is identified as Brian Kernighan's way in https://graphics.stanford.edu/~seander/bithacks.html,
// even if the method was known and used before his implementation, according to the same source.

/// Counts the number of set bits in the byte-sized block of bits received as an argument.
/// The number of iterations is equal to the number of bits set.
pub fn count_set_bits_byte(mut bit_block: u8) -> u8 {
    let mut bit_count = 0;
    while bit_block != 0 {
        bit_block &= bit_block - 1;
        bit_count += 1;
    }
    bit_count
}

pub fn count_set_bits_tetrabyte(mut bit_block: u32) -> u32 {
    let mut bit_count = 0;
    while bit_block != 0 {
        bit_block &= bit_block - 1;
        bit_count += 1;
    }
    bit_count
}

pub fn count_set_bits_hexadecabyte(mut bit_block: u128) -> u32 {
    let mut bit_count = 0;
    while bit_block != 0 {
        bit_block &= bit_block - 1;
        bit_count += 1;
    }
    bit_count
}

#[cfg(feature = "printers")]
mod printers {
    use super::*;

    impl APBFSlice {
        #[allow(dead_code)]
        pub fn data_to_string(&self) -> String {
            let data_as_strings: Vec<String> =
                self.data.iter().map(|b| format!("{:08b}", b)).collect();
            data_as_strings.join(" ")
        }

        #[allow(dead_code)]
        pub fn data_to_lines(&self) -> Vec<String> {
            let mut data_lines = Vec::new();
            let mut line_segments = Vec::with_capacity(8);

            for (i, byte) in self.data.iter().enumerate() {
                line_segments.push(format!("{:08b}", byte));
                if i % 8 == 0 {
                    data_lines.push(line_segments.join(" "));
                    line_segments.clear();
                }
            }
            // If bytes are not aligned to 64 bits, there might be extra bytes to represent
            if !line_segments.is_empty() {
                data_lines.push(line_segments.join(" "));
            }

            data_lines
        }

        pub fn print_data(&self) {
            let data_lines = self.data_to_lines();
            if data_lines.is_empty() {
                println!("{: <10} : <empty>", "Data");
            } else {
                let (head, tail) = data_lines.split_at(1);
                println!("{: <10} : {}", "Data", head[0]);
                for dl in tail {
                    println!("{: <12} {}", "", dl);
                }
            }
        }

        #[allow(dead_code)]
        pub fn print_basic_info(&self) {
            self.print_info(false);
        }

        #[allow(dead_code)]
        pub fn print_full_info(&self) {
            self.print_info(true);
        }

        /// Shows info about the slice.
        fn print_info(&self, print_data: bool) {
            println!("{: <10} : {} bits", "Size", self.size);
            println!("{: <10} : {} bits", "Set bits", self.set_bits);
            println!("{: <10} : {}", "Hash index", self.hash_index);
            if print_data {
                self.print_data();
            }
        }
    }

    impl APBF {
        pub fn print_slice_data(&self) {
            let mut slice_cursor = self.base_slice_idx;
            let mut slices_left = self.num_slices;
            println!("{: >4} {: >4} {: >4} {: <4}", "LOG", "PHY", "IDX", "DATA");
            while slices_left > 0 {
                let data_lines = self.slices[slice_cursor].data_to_lines();
                if data_lines.is_empty() {
                    // println!("{: >4} {: >4} {: >4} {}", "-", "-", "-", "<empty>");
                    println!("{: >4} {: >4} {: >4} ", "-", "-", "-");
                } else {
                    let (head, tail) = data_lines.split_at(1);
                    println!(
                        "{: >4} {: >4} {: >4} {}",
                        self.num_slices - slices_left,
                        slice_cursor,
                        self.slices[slice_cursor].hash_index,
                        head[0]
                    );
                    for dl in tail {
                        println!("{: <14} {}", "", dl);
                    }
                }
                slice_cursor = (slice_cursor + 1).rem_euclid(self.num_slices);
                slices_left -= 1;
            }
        }

        #[allow(dead_code)]
        pub fn print_basic_info(&self) {
            self.print_info(false);
        }

        #[allow(dead_code)]
        pub fn print_full_info(&self) {
            self.print_info(true);
        }

        fn print_info(&self, print_data: bool) {
            println!("> {: <20} : {}", "Base slice index", self.base_slice_idx);
            println!("> {: <20} : {}", "Hash functions", self.num_hashers);
            println!("> {: <20} : {}", "\nOld batches stored", self.num_batches);
            println!("> {: <20} : {}", "Slices", self.num_slices);
            println!("> {: <20} : {}", "Elements inserted", self.inserted);
            if print_data {
                println!("> {: <20} :", "Slice data");
                self.print_slice_data();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    fn single_insertion() {
        let mut apbf = APBF::create_high_level(1, 32, 0, false, true);
        let test_string = "Test";
        // APBF is empty
        assert_eq!(apbf.inserted, 0);
        // False positive probability is 0 before while empty
        assert_eq!(apbf.expected_fp_probabilty(), 0_f64);
        let res = apbf.checked_insert(test_string.as_bytes());
        // APBF item count has increased to 1
        assert_eq!(apbf.inserted, 1);
        // False positive probability is greater than 0 after an insertion
        assert!(apbf.expected_fp_probabilty() > 0_f64);
        // Item is new in the current generation
        assert_eq!(res.0, true);
    }

    #[test]
    fn same_insertion() {
        let mut apbf = APBF::create_high_level(1, 32, 0, false, true);
        let test_string = "Test";
        let fst_res = apbf.checked_insert(test_string.as_bytes());
        let snd_res = apbf.checked_insert(test_string.as_bytes());
        // Item was new in the current generation
        assert_eq!(fst_res.0, true);
        // Item was already in the current generation
        assert_eq!(snd_res.0, false);
        if cfg!(feature = "count_if_new_in_cur_gen") {
            assert_eq!(apbf.inserted, 1);
        } else {
            assert_eq!(apbf.inserted, 2);
        }
    }

    #[test]
    fn multiple_insertion() {
        let mut apbf = APBF::create_high_level(1, 32, 0, false, true);
        let qarray = [
            "Test 1".as_bytes(),
            "Test 2".as_bytes(),
            "Test 3".as_bytes(),
            "Test 1".as_bytes(),
        ];
        let res = apbf
            .multiple_insert(&qarray)
            .iter()
            .map(|t| t.0)
            .collect::<Vec<bool>>();
        // The last item will already be known and so its insertion will return false
        assert_eq!(res, vec![true, true, true, false]);
    }

    #[test]
    fn query_after_shift() {
        let mut apbf = APBF::create_high_level(1, 32, 0, false, true);
        let last_slice_idx = apbf.num_slices - 1;
        let test_string = "Test";
        let test_string_bytes = test_string.as_bytes();
        apbf.checked_insert(test_string_bytes);
        apbf.shift_slices();
        // Check that the new base slice is the previous last slice in the circular buffer
        assert_eq!(apbf.base_slice_idx, last_slice_idx);
        // Check that even if outside the first k slices, the item can still be found, as all the
        // k slices used for its insertion still exist in the APBF
        let res = apbf.query(test_string_bytes);
        assert_eq!(res, true);
    }

    #[test]
    fn multiple_query() {
        let mut apbf = APBF::create_high_level(1, 32, 0, false, true);
        let qarray = [
            "Test 1".as_bytes(),
            "Test 2".as_bytes(),
            "Test 3".as_bytes(),
            "Test 1".as_bytes(),
        ];
        apbf.multiple_insert(&qarray);
        let res = apbf.multiple_query(&qarray);
        // The last item will already be in the first k slices and so its insertion will return false
        assert_eq!(res, vec![true, true, true, true]);
    }

    #[test]
    fn state_after_shift() {
        let mut apbf = APBF::create_high_level(3, 4096, 2, false, true);
        // The initial base slice index of an APBF is slice 0
        let mut base_slice_idx: usize = 0;
        assert_eq!(base_slice_idx, 0);
        for e in 0..10_000_u64 {
            apbf.insert(&e.to_be_bytes());
            let last_slice_idx = apbf.log_to_phy_idx(apbf.num_slices - 1);
            // Obtaining the hash index of the slice at index k - 1, which should be the hash index
            // of the new first of the circular buffer, if shifting happens.
            let future_base_hash_index =
                apbf.slices[apbf.log_to_phy_idx(apbf.num_hashers - 1)].hash_index;
            let shifts = apbf.check_and_shift();
            // Checks that at most, only one shift will be performed if inserting elements one by one.
            assert!(shifts == 0 || shifts == 1);
            if shifts > 0 {
                // Checking if the new base slice's index is of the previous last slice of the circular
                // buffer.
                let new_base_slice_idx = (base_slice_idx as isize - shifts as isize)
                    .rem_euclid(apbf.num_slices as isize)
                    as usize;
                // Checking if the new base slice index of the slice is updating as expected.
                assert_eq!(new_base_slice_idx, apbf.base_slice_idx);
                assert_eq!(new_base_slice_idx, last_slice_idx);
                base_slice_idx = new_base_slice_idx;
                // Checking if after shifting, the current_gen of every slice is empty.
                for slice in apbf.slices.iter() {
                    assert_eq!(slice.count_set_current_gen_bits(), 0);
                }
                // Checking if some fields are being cleared as expected.
                assert_eq!(apbf.slices[new_base_slice_idx].set_bits, 0);
                assert_eq!(apbf.slices[new_base_slice_idx].count_set_data_bits(), 0);
                // Checking that the slice following the base has not been cleared, preserving the
                // content from the previous generation.
                assert_ne!(apbf.slices[apbf.log_to_phy_idx(1)].count_set_data_bits(), 0);
                // Checking if the hash index of the new base slice has the expected value.
                assert_eq!(
                    apbf.slices[new_base_slice_idx].hash_index,
                    future_base_hash_index
                );
            }
        }
        for slice in apbf.slices.iter() {
            assert_eq!(slice.set_bits, slice.count_set_data_bits());
        }
    }

    #[test]
    fn multi_shift() {
        let mut apbf = APBF::create_high_level(3, 4096, 2, false, true);
        assert_eq!(apbf.base_slice_idx, 0);
        // Inserting some elements, just so the slices have some bits set.
        for e in 0..256_u32 {
            apbf.insert(&e.to_be_bytes());
        }
        let shifts = 3;
        apbf.shift_slices_times(shifts);
        let expected_base_idx = (apbf.num_slices - shifts).rem_euclid(apbf.num_slices);
        assert_eq!(apbf.base_slice_idx, expected_base_idx);
        for offset in 0..shifts {
            let idx = apbf.log_to_phy_idx(offset);
            let slice = &apbf.slices[idx];
            assert_eq!(slice.set_bits, 0);
            assert_eq!(slice.count_set_data_bits(), 0);
            assert_eq!(slice.count_set_current_gen_bits(), 0);
        }
        let mut hash_indices = HashSet::with_capacity(apbf.num_hashers);
        for log_idx in 0..apbf.num_hashers {
            hash_indices.insert(apbf.slices[apbf.log_to_phy_idx(log_idx)].hash_index);
        }
        // Every active slice (within the first k of the filter) should have a different hash index
        assert_eq!(hash_indices.len(), apbf.num_hashers);
    }

    #[test]
    fn basic_multi_shift_equality() {
        let mut apbf1 = APBF::create_high_level(3, 4096, 2, false, true);
        let mut apbf2 = APBF::create_high_level(3, 4096, 2, false, true);
        for e in 0..8192_u32 {
            apbf1.insert(&e.to_be_bytes());
            apbf2.insert(&e.to_be_bytes());
            let ns1 = apbf1.calc_needed_shifts();
            let ns2 = apbf2.calc_needed_shifts();
            assert_eq!(ns1, ns2);
            apbf1.shift_slices_times(ns1);
            for _ in 0..ns2 {
                apbf2.shift_slices();
            }
            assert_eq!(apbf1.base_slice_idx, apbf2.base_slice_idx);
        }
        for i in 0..apbf1.num_slices {
            let apbf1_slice = &apbf1.slices[i];
            let apbf2_slice = &apbf2.slices[i];
            assert_eq!(apbf1_slice.set_bits, apbf2_slice.set_bits);
            assert_eq!(apbf1_slice.hash_index, apbf2_slice.hash_index);
            assert_eq!(
                apbf1_slice.count_set_data_bits(),
                apbf2_slice.count_set_data_bits()
            );
            assert_eq!(apbf1_slice.count_set_data_bits(), apbf1_slice.set_bits);
            assert_eq!(apbf2_slice.count_set_data_bits(), apbf2_slice.set_bits);
            assert_eq!(
                apbf1_slice.count_set_current_gen_bits(),
                apbf2_slice.count_set_current_gen_bits()
            );
        }
    }
}
