package org.apache.spark.ml.image.core

import org.opencv.core.{Mat, Rect}

import scala.util.Random

class Cropper(cH: Int, cW: Int, cropperMethod: String = "center")
    extends ProcessStep[cMat, cMat] {

  override def apply(prev: cMat): cMat = {
    val width = prev.width()
    val height = prev.height()

    val (startH, startW) = cropperMethod match {
      case "random" =>
        (math.ceil(new Random().nextInt(height - cH)).toInt,
          math.ceil(new Random().nextInt(width - cW)).toInt)
      case "center" =>
        ((height - cH) / 2, (width - cW) / 2)
    }

    val rect = new Rect(startW, startH, cW, cH)
    new cMat(new Mat(prev, rect))

  }
}

object Cropper {
  def apply(cropHeight: Int, cropWidth: Int,
            cropperMethod: String = "center"): Cropper =
    new Cropper(cropWidth, cropHeight, cropperMethod)
}

