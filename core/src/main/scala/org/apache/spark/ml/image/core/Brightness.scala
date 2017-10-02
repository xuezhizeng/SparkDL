package org.apache.spark.ml.image.core

class Brightness (delta: Float, inplace: Boolean = true) extends ProcessStep[cMat, cMat] {
  override def apply(prev: cMat): cMat = {
      Brightness.transform(prev, prev, delta)
  }
}

object Brightness {
  def apply(delta: Float, inplace: Boolean = true): Brightness = new Brightness(delta, inplace)

  def transform(inMat: cMat, outMat: cMat, delta: Float): cMat = {
    if (delta != 0) {
      inMat.convertTo(outMat, -1, 1, delta)
      outMat
    } else {
      inMat
    }
  }
}
