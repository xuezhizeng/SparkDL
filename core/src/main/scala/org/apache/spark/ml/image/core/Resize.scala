package org.apache.spark.ml.image.core

import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

class Resize(resizeHeight: Int, resizeWidth: Int) extends ProcessStep[cMat, cMat] {

  override def apply(prev: cMat): cMat = {
    var resized = new cMat()
    val sz = new Size(resizeWidth, resizeHeight)
    Imgproc.resize(prev, resized, sz)
    resized
  }
}

object Resize {
  def apply(resizeHeight: Int, resizeWidth: Int): Resize = new Resize(resizeHeight, resizeWidth)
}


