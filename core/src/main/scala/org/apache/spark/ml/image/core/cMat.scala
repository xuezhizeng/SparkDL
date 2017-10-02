package org.apache.spark.ml.image.core

import java.io.File

import com.intel.analytics.OpenCV
import org.opencv.core.{Mat, MatOfByte, Rect}
import org.opencv.imgcodecs.Imgcodecs

class cMat extends Mat with Serializable {

  def this(mat: Mat) = {
    this()
    mat.copyTo(this)
  }

}

object cMat {


}
