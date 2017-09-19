package org.apache.spark.ml.image.core

import scala.util.Random

object BGRImageCropper {
  def apply(cropWidth: Int, cropHeight: Int,
            cropperMethod: String = "center"): BGRImageCropper =
    new BGRImageCropper(cropWidth, cropHeight, cropperMethod)
}


class BGRImageCropper(cW: Int, cH: Int, cropperMethod: String)
    extends ProcessStep[(Array[Float], Array[Int]), (Array[Float], Array[Int])] {

  override def apply(prev: (Array[Float], Array[Int])): (Array[Float], Array[Int]) = {
    val width = prev._2(1)
    val height = prev._2(2)
    val numChannel = prev._2(0)
    val source = prev._1

    val (startH, startW) = cropperMethod match {
      case "random" =>
        (math.ceil(new Random().nextInt(height - cH)).toInt,
          math.ceil(new Random().nextInt(width - cW)).toInt)
      case "center" =>
        ((height - cH) / 2, (width - cW) / 2)
    }
    val startIndex = (startW + startH * width) * 3
    val frameLength = cW * cW
    val target = new Array[Float](numChannel * cW * cH)
    var i = 0
    while (i < frameLength) {
      target(i * 3 + 2) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3 + 2)
      target(i * 3 + 1) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3 + 1)
      target(i * 3) =
        source(startIndex + ((i / cW) * width + (i % cW)) * 3)
      i += 1
    }
    require(target.length == cW * cH * numChannel)
    (target, Array(numChannel, cW, cH))
  }
}
