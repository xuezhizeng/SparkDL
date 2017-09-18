package org.apache.spark.ml.image.core

import scala.collection.Iterator
import scala.reflect.ClassTag

trait TransformStep[A, B] extends Serializable {
  def apply(prev: A): B

  def -> [C](other: TransformStep[B, C]): TransformStep[A, C] = {
    new ChainedTransformerStep(this, other)
  }

}

class ChainedTransformerStep[A, B, C](first: TransformStep[A, B], last: TransformStep[B, C])
  extends TransformStep[A, C] {
  override def apply(prev: A): C = {
    last(first(prev))
  }
}