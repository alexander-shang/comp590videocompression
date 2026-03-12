use std::env;
use std::path::PathBuf;


use ffmpeg_sidecar::command::FfmpegCommand;
use workspace_root::get_workspace_root;


use std::fs::File;
use std::io::BufReader;
use std::io::{BufWriter, Write};


use bitbit::BitReader;
use bitbit::BitWriter;
use bitbit::MSB;


use toy_ac::decoder::Decoder;
use toy_ac::encoder::Encoder;
use toy_ac::symbol_model::VectorCountSymbolModel;


use ffmpeg_sidecar::event::StreamTypeSpecificData::Video;


fn main() -> Result<(), Box<dyn std::error::Error>> {
   // Make sure ffmpeg is installed
   ffmpeg_sidecar::download::auto_download().unwrap();


   // Command line options
   // -verbose, -no_verbose                Default: -no_verbose
   // -report, -no_report                  Default: -report
   // -check_decode, -no_check_decode      Default: -no_check_decode
   // -skip_count n                        Default: -skip_count 0
   // -count n                             Default: -count 10
   // -in file_path                        Default: bourne.mp4 in data subdirectory of workplace
   // -out file_path                       Default: out.dat in data subdirectory of workplace


   // Set up default values of options
   let mut verbose = true;
   let mut report = true;
   let mut check_decode = true;
   let mut skip_count = 0;
   let mut count = 10;


   let mut data_folder_path = get_workspace_root();
   data_folder_path.push("data");


   let mut input_file_path = data_folder_path.join("drewgooden.mp4");
   let mut output_file_path = data_folder_path.join("out.dat");


   parse_args(
       &mut verbose,
       &mut report,
       &mut check_decode,
       &mut skip_count,
       &mut count,
       &mut input_file_path,
       &mut output_file_path,
   );


   // Run an FFmpeg command to decode video from inptu_file_path
   // Get output as grayscale (i.e., just the Y plane)


   let mut iter = FfmpegCommand::new() // <- Builder API like `std::process::Command`
       .input(input_file_path.to_str().unwrap())
       .format("rawvideo")
       .pix_fmt("gray8")
       .output("-")
       .spawn()? // <- Ordinary `std::process::Child`
       .iter()?; // <- Blocking iterator over logs and output


   // Figure out geometry of frame.
   let mut width = 0;
   let mut height = 0;


   let metadata = iter.collect_metadata()?;
   for i in 0..metadata.output_streams.len() {
       match &metadata.output_streams[i].type_specific_data {
           Video(vid_stream) => {
               width = vid_stream.width;
               height = vid_stream.height;


               if verbose {
                   println!(
                       "Found video stream at output stream index {} with dimensions {} x {}",
                       i, width, height
                   );
               }
               break;
           }
           _ => (),
       }
   }
   assert!(width != 0);
   assert!(height != 0);


   // Set up initial prior frame as uniform medium gray (y = 128)
   let mut prior_frame = vec![128 as u8; (width * height) as usize];


   let output_file = match File::create(&output_file_path) {
       Err(_) => panic!("Error opening output file"),
       Ok(f) => f,
   };


   // Setup bit writer and arithmetic encoder.


   let mut buf_writer = BufWriter::new(output_file);
   let mut bw = BitWriter::new(&mut buf_writer);


   let mut enc = Encoder::new();


   // Set up arithmetic coding contexts.
   // We use 8 contexts selected by local activity level (spatial gradient + temporal diff).
   // This stays well within the 256-context limit.
   const NUM_CONTEXTS: usize = 8;
   let mut pixel_difference_pdfs: Vec<VectorCountSymbolModel<i32>> = (0..NUM_CONTEXTS)
       .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
       .collect();


   // Buffer to hold decoded pixel values for the current frame as we scan,
   // so the predictor can use already-encoded neighbors.
   let n_pixels = (width * height) as usize;
   let mut current_frame_buf = vec![0i32; n_pixels];


   // Helper: median of three values (used for spatial prediction).
   let median3 = |a: i32, b: i32, c: i32| -> i32 {
       a + b + c - a.min(b).min(c) - a.max(b).max(c)
   };


   // Helper: given already-scanned neighbors, compute a blended spatial+temporal prediction.
   let predict = |buf: &[i32], prior: &[u8], r: u32, c: u32| -> i32 {
       let idx  = (r * width + c) as usize;
       let left = if c > 0           { buf[idx - 1] }                   else { prior[idx] as i32 };
       let top  = if r > 0           { buf[idx - width as usize] }       else { prior[idx] as i32 };
       let tl   = if r > 0 && c > 0  { buf[idx - width as usize - 1] }   else { prior[idx] as i32 };
       let temporal = prior[idx] as i32;
       let spatial  = median3(left, top, left + top - tl); // PNG-style predictor
       (spatial + temporal) / 2
   };


   // Helper: select context index based on local activity.
   let select_ctx = |buf: &[i32], prior: &[u8], r: u32, c: u32| -> usize {
       let idx      = (r * width + c) as usize;
       let left     = if c > 0 { buf[idx - 1] }                 else { 128 };
       let top      = if r > 0 { buf[idx - width as usize] }     else { 128 };
       let temporal = prior[idx] as i32;
       let activity = (left - top).abs().max((left - temporal).abs());
       match activity {
           0..=1    => 0,
           2..=4    => 1,
           5..=9    => 2,
           10..=19  => 3,
           20..=39  => 4,
           40..=79  => 5,
           80..=127 => 6,
           _        => 7,
       }
   };


   // Process frames
   for frame in iter.filter_frames() {
       if frame.frame_num < skip_count {
           if verbose {
               println!("Skipping frame {}", frame.frame_num);
           }
       } else if frame.frame_num < skip_count + count {
           let current_frame: Vec<u8> = frame.data;


           let bits_written_at_start = enc.bits_written();


           for r in 0..height {
               for c in 0..width {
                   let idx = (r * width + c) as usize;


                   let pred     = predict(&current_frame_buf, &prior_frame, r, c);
                   let actual   = current_frame[idx] as i32;
                   let residual = ((actual - pred) + 256) % 256;
                   let ctx      = select_ctx(&current_frame_buf, &prior_frame, r, c);


                   enc.encode(&residual, &pixel_difference_pdfs[ctx], &mut bw);
                   pixel_difference_pdfs[ctx].incr_count(&residual);


                   current_frame_buf[idx] = actual;
               }
           }


           prior_frame = current_frame;


           let bits_written_at_end = enc.bits_written();


           if verbose {
               println!(
                   "frame: {}, compressed size (bits): {}",
                   frame.frame_num,
                   bits_written_at_end - bits_written_at_start
               );
           }
       } else {
           break;
       }
   }


   // Tie off arithmetic encoder and flush to file.
   enc.finish(&mut bw)?;
   bw.pad_to_byte()?;
   buf_writer.flush()?;


   // Decompress and check for correctness.
   if check_decode {
       let output_file = match File::open(&output_file_path) {
           Err(_) => panic!("Error opening output file"),
           Ok(f) => f,
       };
       let mut buf_reader = BufReader::new(output_file);
       let mut br: BitReader<_, MSB> = BitReader::new(&mut buf_reader);


       let iter = FfmpegCommand::new()
           .input(input_file_path.to_str().unwrap())
           .format("rawvideo")
           .pix_fmt("gray8")
           .output("-")
           .spawn()?
           .iter()?;


       let mut dec = Decoder::new();


       let mut dec_pdfs: Vec<VectorCountSymbolModel<i32>> = (0..NUM_CONTEXTS)
           .map(|_| VectorCountSymbolModel::new((0..=255).collect()))
           .collect();


       let mut prior_frame = vec![128u8; n_pixels];
       let mut current_frame_buf = vec![0i32; n_pixels];


       'outer_loop:
       for frame in iter.filter_frames() {
           if frame.frame_num < skip_count + count {
               if verbose {
                   print!("Checking frame: {} ... ", frame.frame_num);
               }


               let current_frame: Vec<u8> = frame.data;


               for r in 0..height {
                   for c in 0..width {
                       let idx = (r * width + c) as usize;


                       let pred     = predict(&current_frame_buf, &prior_frame, r, c);
                       let ctx      = select_ctx(&current_frame_buf, &prior_frame, r, c);
                       let residual = dec.decode(&dec_pdfs[ctx], &mut br).to_owned();
                       dec_pdfs[ctx].incr_count(&residual);


                       let reconstructed = ((pred + residual) % 256 + 256) % 256;
                       current_frame_buf[idx] = reconstructed;


                       if reconstructed != current_frame[idx] as i32 {
                           println!(
                               " error at ({}, {}), should decode {}, got {}",
                               c, r, current_frame[idx], reconstructed
                           );
                           println!("Abandoning check of remaining frames");
                           break 'outer_loop;
                       }
                   }
               }


               if verbose { println!("correct."); }
               prior_frame = current_frame;
           } else {
               break 'outer_loop;
           }
       }
   }


   // Emit report
   if report {
       println!(
           "{} frames encoded, average size (bits): {}, compression ratio: {:.2}",
           count,
           enc.bits_written() / count as u64,
           (width * height * 8 * count) as f64 / enc.bits_written() as f64
       )
   }


   Ok(())
}


fn parse_args(
   verbose: &mut bool,
   report: &mut bool,
   check_decode: &mut bool,
   skip_count: &mut u32,
   count: &mut u32,
   input_file_path: &mut PathBuf,
   output_file_path: &mut PathBuf,
) -> () {
   let mut args = env::args().skip(1);


   while let Some(arg) = args.next() {
       if arg == "-verbose" {
           *verbose = true;
       } else if arg == "-no_verbose" {
           *verbose = false;
       } else if arg == "-report" {
           *report = true;
       } else if arg == "-no_report" {
           *report = false;
       } else if arg == "-check_decode" {
           *check_decode = true;
       } else if arg == "-no_check_decode" {
           *check_decode = false;
       } else if arg == "-skip_count" {
           match args.next() {
               Some(skip_count_string) => {
                   *skip_count = skip_count_string.parse::<u32>().unwrap();
               }
               None => {
                   panic!("Expected count after -skip_count option");
               }
           }
       } else if arg == "-count" {
           match args.next() {
               Some(count_string) => {
                   *count = count_string.parse::<u32>().unwrap();
               }
               None => {
                   panic!("Expected count after -count option");
               }
           }
       } else if arg == "-in" {
           match args.next() {
               Some(input_file_path_string) => {
                   *input_file_path = PathBuf::from(input_file_path_string);
               }
               None => {
                   panic!("Expected input file name after -in option");
               }
           }
       } else if arg == "-out" {
           match args.next() {
               Some(output_file_path_string) => {
                   *output_file_path = PathBuf::from(output_file_path_string);
               }
               None => {
                   panic!("Expected output file name after -out option");
               }
           }
       }
   }
}





