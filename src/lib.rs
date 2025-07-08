#![deny(clippy::all)]

use napi::bindgen_prelude::*;
use napi_derive::napi;
use opencv::{
  core::{self, Point, Size, Vector},
  imgcodecs, imgproc,
  prelude::*,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
#[napi(object)]
pub struct MatchOptions {
  pub threshold: Option<f64>,
  pub scale: Option<f64>,
}

#[derive(Serialize)]
#[napi(object)]
pub struct MatchResult {
  pub found: bool,
  pub x: Option<i32>,
  pub y: Option<i32>,
  pub confidence: f64,
}

#[napi]
pub fn template_match(
  haystack_buffer: Buffer,
  needle_buffer: Buffer,
  options: Option<MatchOptions>,
) -> Result<MatchResult> {
  let threshold = options.as_ref().and_then(|o| o.threshold).unwrap_or(0.8);
  let scale = options.as_ref().and_then(|o| o.scale).unwrap_or(1.0);

  let haystack_data: Vec<u8> = haystack_buffer.to_vec();
  let needle_data: Vec<u8> = needle_buffer.to_vec();

  let haystack_vector = Vector::<u8>::from_iter(haystack_data);
  let needle_vector = Vector::<u8>::from_iter(needle_data);

  let haystack = imgcodecs::imdecode(&haystack_vector, imgcodecs::IMREAD_COLOR)
    .map_err(|e| Error::new(Status::GenericFailure, format!("解码大图失败: {}", e)))?;

  let needle = imgcodecs::imdecode(&needle_vector, imgcodecs::IMREAD_COLOR)
    .map_err(|e| Error::new(Status::GenericFailure, format!("解码模板图失败: {}", e)))?;

  if haystack.empty() || needle.empty() {
    return Err(Error::new(Status::InvalidArg, "图片数据为空".to_string()));
  }

  let haystack_size = haystack
    .size()
    .map_err(|e| Error::new(Status::GenericFailure, format!("获取大图尺寸失败: {}", e)))?;

  let new_size = Size::new(
    (needle.cols() as f64 * scale) as i32,
    (needle.rows() as f64 * scale) as i32,
  );

  let mut resized_needle = Mat::default();
  imgproc::resize(
    &needle,
    &mut resized_needle,
    new_size,
    0.0,
    0.0,
    imgproc::INTER_LINEAR,
  )
  .map_err(|e| Error::from_reason(e.to_string()))?;

  let needle_size = resized_needle
    .size()
    .map_err(|e| Error::from_reason(e.to_string()))?;

  if haystack_size.width < needle_size.width || haystack_size.height < needle_size.height {
    return Err(Error::new(
      Status::InvalidArg,
      "模板图尺寸大于大图，无法匹配".to_string(),
    ));
  }

  let result_cols = haystack_size.width - needle_size.width + 1;
  let result_rows = haystack_size.height - needle_size.height + 1;

  let mut result = Mat::default();

  unsafe {
    result
      .create_size(Size::new(result_cols, result_rows), core::CV_32FC1)
      .map_err(|e| Error::new(Status::GenericFailure, format!("创建结果矩阵失败: {}", e)))?;
  }

  imgproc::match_template(
    &haystack,
    &resized_needle,
    &mut result,
    imgproc::TM_CCOEFF_NORMED,
    &Mat::default(),
  )
  .map_err(|e| Error::new(Status::GenericFailure, format!("模板匹配失败: {}", e)))?;

  let mut min_val = 0.0;
  let mut max_val = 0.0;
  let mut min_loc = Point::default();
  let mut max_loc = Point::default();

  core::min_max_loc(
    &result,
    Some(&mut min_val),
    Some(&mut max_val),
    Some(&mut min_loc),
    Some(&mut max_loc),
    &Mat::default(),
  )
  .map_err(|e| Error::new(Status::GenericFailure, format!("查找最值失败: {}", e)))?;

  let matched = max_val >= threshold;

  Ok(MatchResult {
    found: matched,
    x: if matched { Some(max_loc.x) } else { None },
    y: if matched { Some(max_loc.y) } else { None },
    confidence: max_val,
  })
}
