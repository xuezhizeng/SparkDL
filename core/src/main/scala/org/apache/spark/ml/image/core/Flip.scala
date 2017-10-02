package org.apache.spark.ml.image.core

import org.opencv.core.{Core, Size}
import org.opencv.imgproc.Imgproc

class Flip(direction: Int) extends ProcessStep[cMat, cMat] {

  override def apply(prev: cMat): cMat = {
    Core.flip(prev, prev, direction)
    prev
  }
}

object Flip {
  def apply(direction: Int): Flip = new Flip(direction)

  val HORIZONTAL_FLIP = 0
  val VERTICAL_FLIP = 1
  val BOTH_FLIP = -1
}

