package com.spotify.annoy;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * Read-only Approximate Nearest Neighbor Index which queries databases created by annoy.
 */
public class ANNIndex implements AnnoyIndex {

	private final ArrayList<Long> roots;
	private MappedByteBuffer[] buffers;

	private final int DIMENSION, MIN_LEAF_SIZE;
	private final IndexType INDEX_TYPE;
	private final int INDEX_TYPE_OFFSET;

	// size of C structs in bytes (initialized in init)
	private final int K_NODE_HEADER_STYLE;
	private final long NODE_SIZE;

	private final int INT_SIZE = 4;
	private final int FLOAT_SIZE = 4;
	private final int MAX_NODES_IN_BUFFER;
	private final int BLOCK_SIZE;
	private RandomAccessFile memoryMappedFile;
	private final int nItems;

	/**
	 * Construct and load an Annoy index of a specific type (euclidean / angular).
	 *
	 * @param dimension
	 *            dimensionality of tree, e.g. 40
	 * @param filename
	 *            filename of tree
	 * @param indexType
	 *            type of index
	 * @throws IOException
	 *             if file can't be loaded
	 */
	public ANNIndex(final int dimension, final String filename, IndexType indexType) throws IOException {
		this(dimension, filename, indexType, 0);
	}

	private ANNIndex(final int dimension, final String filename, IndexType indexType, final int blockSize) throws IOException {
		DIMENSION = dimension;
		INDEX_TYPE = indexType;
		INDEX_TYPE_OFFSET = INDEX_TYPE.getOffset();
		K_NODE_HEADER_STYLE = INDEX_TYPE.getkNodeHeaderStyle();
		// we can store up to MIN_LEAF_SIZE children in leaf nodes (we put
		// them where the separating plane normally goes)
		this.MIN_LEAF_SIZE = DIMENSION + 2;
		this.NODE_SIZE = K_NODE_HEADER_STYLE + FLOAT_SIZE * DIMENSION;
		this.MAX_NODES_IN_BUFFER = (int) (blockSize == 0 ? Integer.MAX_VALUE / NODE_SIZE : blockSize * NODE_SIZE);
		BLOCK_SIZE = (int) (this.MAX_NODES_IN_BUFFER * NODE_SIZE);
		roots = new ArrayList<>();
		nItems = load(filename);
	}

	private int load(final String filename) throws IOException {
		memoryMappedFile = new RandomAccessFile(filename, "r");
		long fileSize = memoryMappedFile.length();
		if (fileSize == 0L) {
			throw new IOException("Index is a 0-byte file?");
		}

		int numNodes = (int) (fileSize / NODE_SIZE);
		int buffIndex = (numNodes - 1) / MAX_NODES_IN_BUFFER;
		int rest = (int) (fileSize % BLOCK_SIZE);
		int blockSize = (rest > 0 ? rest : BLOCK_SIZE);
		// Two valid relations between dimension and file size:
		// 1) rest % NODE_SIZE == 0 makes sure either everything fits into buffer or rest is a
		// multiple of NODE_SIZE;
		// 2) (file_size - rest) % NODE_SIZE == 0 makes sure everything else is a multiple of
		// NODE_SIZE.
		if (rest % NODE_SIZE != 0 || (fileSize - rest) % NODE_SIZE != 0) {
			throw new RuntimeException("ANNIndex initiated with wrong dimension size");
		}
		long position = fileSize - blockSize;
		buffers = new MappedByteBuffer[buffIndex + 1];
		boolean process = true;
		int m = -1;
		long index = fileSize;
		while (position >= 0) {
			MappedByteBuffer annBuf = memoryMappedFile.getChannel().map(FileChannel.MapMode.READ_ONLY, position, blockSize);
			annBuf.order(ByteOrder.LITTLE_ENDIAN);

			buffers[buffIndex--] = annBuf;

			for (int i = blockSize - (int) NODE_SIZE; process && i >= 0; i -= NODE_SIZE) {
				index -= NODE_SIZE;
				int k = annBuf.getInt(i); // node[i].n_descendants
				if (m == -1 || k == m) {
					roots.add(index);
					m = k;
				} else {
					process = false;
				}
			}
			blockSize = BLOCK_SIZE;
			position -= blockSize;
		}
		int n1 = getIntInAnnBuf(roots.get(0) + INDEX_TYPE_OFFSET);
		int n2 = getIntInAnnBuf(roots.get(roots.size() - 1) + INDEX_TYPE_OFFSET);
		if (n1 == n2) {
			roots.remove(roots.size() - 1);
		}
		return m;
	}

	private float getFloatInAnnBuf(long pos) {
		int b = (int) (pos / BLOCK_SIZE);
		int f = (int) (pos % BLOCK_SIZE);
		return buffers[b].getFloat(f);
	}

	private int getIntInAnnBuf(long pos) {
		int b = (int) (pos / BLOCK_SIZE);
		int i = (int) (pos % BLOCK_SIZE);
		return buffers[b].getInt(i);
	}

	@Override
	public void getNodeVector(final long nodeOffset, float[] v) {
		MappedByteBuffer nodeBuf = buffers[(int) (nodeOffset / BLOCK_SIZE)];
		int offset = (int) ((nodeOffset % BLOCK_SIZE) + K_NODE_HEADER_STYLE);
		for (int i = 0; i < DIMENSION; i++) {
			v[i] = nodeBuf.getFloat(offset + i * FLOAT_SIZE);
		}
	}

	@Override
	public void getItemVector(int itemIndex, float[] v) {
		getNodeVector(itemIndex * NODE_SIZE, v);
	}

	private float getNodeBias(final long nodeOffset) { // euclidean-only
		return getFloatInAnnBuf(nodeOffset + 4);
	}

	private float getDotFactor(final long nodeOffset) { // dot-only
		return getFloatInAnnBuf(nodeOffset + 12);
	}

	public final float[] getItemVector(final int itemIndex) {
		return getNodeVector(itemIndex * NODE_SIZE);
	}

	public float[] getNodeVector(final long nodeOffset) {
		float[] v = new float[DIMENSION];
		getNodeVector(nodeOffset, v);
		return v;
	}

	public static float dot(final float[] u, final float[] v, final int f) {
		float d = 0;
		for (int i = 0; i < f; i++)
			d += u[i] * v[i];
		return d;
	}

	public static float cosineMargin(final float[] u, final float[] v, final int f) {
		return dot(u, v, f);
	}

	public static float dotMargin(final float[] u, final float[] v, final float norm, final int f) {
		return dot(u, v, f) + norm * norm;
	}

	public static float euclideanMargin(final float[] u, final float[] v, final float bias, final int f) {
		float d = bias;
		for (int i = 0; i < f; i++)
			d += u[i] * v[i];
		return d;
	}

	public static float manhattanMargin(final float[] u, final float[] v, final float bias, final int f) {
		float d = bias;
		for (int i = 0; i < f; i++)
			d += u[i] * v[i];
		return d;
	}

	private static float normalizedDistance(final float d) {
		return (float) Math.sqrt(d < 0f ? 0f : d);
	}

	public static float angularDistance(final float[] u, final float[] v, final int f, final float uDot, final float vDot) {
		float pq = dot(u, v, f);
		float ppqq = uDot * vDot;
		if (ppqq > 0) {
			return normalizedDistance((float) (2.0 - 2.0 * pq / Math.sqrt(ppqq)));
		} else {
			return normalizedDistance(2f);
		}
	}

	private static float euclideanDistance(final float[] u, final float[] v, final int f) {
		float sum = 0;
		for (int i = 0; i < f; i++) {
			final float dp = u[i] - v[i];
			sum += dp * dp;
		}
		return (float) Math.sqrt(sum < 0f ? 0f : sum);
	}

	private static float manhattanDistance(final float[] u, final float[] v, final int f) {
		float sum = 0;
		for (int i = 0; i < f; i++) {
			sum += Math.abs(u[i] - v[i]);
		}
		return sum < 0f ? 0f : sum;
	}

	/**
	 * Closes this stream and releases any system resources associated with it. If the stream is
	 * already closed then invoking this method has no effect.
	 *
	 * <p>
	 * As noted in {@link AutoCloseable#close()}, cases where the close may fail require careful
	 * attention. It is strongly advised to relinquish the underlying resources and to internally
	 * <em>mark</em> the {@code Closeable} as closed, prior to throwing the {@code IOException}.
	 *
	 * @throws IOException
	 *             if an I/O error occurs
	 */
	@Override
	public void close() throws IOException {
		memoryMappedFile.close();
	}

	public class PQEntry implements Comparable<PQEntry> {

		PQEntry(final float margin, final long nodeOffset) {
			this.margin = margin;
			this.nodeOffset = nodeOffset;
		}

		private final float margin;
		private final long nodeOffset;

		public float getMargin() {
			return margin;
		}

		public long getNodeOffset() {
			return nodeOffset;
		}

		@Override
		public int compareTo(final PQEntry o) {
			// Optimized under the assumption that margin will generally not be equal and therefore
			// one of the first two checks will usually result in a return statement
			if (this.margin < o.margin) {
				return -1;
			} else if (this.margin > o.margin) {
				return 1;
			} else {
				if (!(o.margin < this.margin) && this.nodeOffset < o.nodeOffset) {
					return -1;
				} else if (this.nodeOffset == o.nodeOffset) {
					return 0;
				} else {
					return 1;
				}
			}

			// Unoptimized code
			// if (this.margin == o.margin && this.nodeOffset == o.nodeOffset) {
			// return 0;
			// } else if (this.margin < o.margin || (!(o.margin < this.margin) && this.nodeOffset <
			// o.nodeOffset)) {
			// return -1;
			// } else {
			// return 1;
			// }
		}

		public boolean equal(final PQEntry o) {
			return this.margin == o.margin && this.nodeOffset == o.nodeOffset;
		}

	}

	private static boolean isZeroVec(float[] v) {
		for (int i = 0; i < v.length; i++)
			if (v[i] != 0)
				return false;
		return true;
	}

	@Override
	public final List<Integer> getNearest(final float[] queryVector, final int nResults) {
		return getNearest(queryVector, nResults, -1);
	}

	public final List<Integer> getNearest(final float[] queryVector, final int nResults, int searchK) {
		List<PQEntry> resultingPQEntries = getNearestPqEntries(queryVector, nResults, searchK);
		ArrayList<Integer> result = new ArrayList<>(nResults);
		for (PQEntry pqEntry : resultingPQEntries) {
			result.add((int) pqEntry.nodeOffset);
		}
		return result;
	}

	@Override
	public final List<PQEntry> getNearestPqEntries(final float[] queryVector, final int nResults) {
		return getNearestPqEntries(queryVector, nResults, -1);
	}

	public List<PQEntry> getNearestPqEntries(final float[] queryVector, final int nResults, int searchK) {
		if (queryVector.length != DIMENSION) {
			throw new RuntimeException(String.format("queryVector must be size of %d, but was %d", DIMENSION, queryVector.length));
		}
		float queryVectorDot = 0f;
		if (INDEX_TYPE == IndexType.ANGULAR) {
			queryVectorDot = dot(queryVector, queryVector, DIMENSION);
		}

		PriorityQueue<PQEntry> pq = new PriorityQueue<>(roots.size() * FLOAT_SIZE, Collections.reverseOrder());
		final float kMaxPriority = Float.POSITIVE_INFINITY;// 1e30f;

		for (long r : roots) {
			pq.add(new PQEntry(kMaxPriority, r));
		}

		if (searchK == -1) {
			searchK = roots.size() * nResults;
		}
		int neighborsCount = 0;// Use this counter since the set size will not be accurate with
								// respect to duplicate entries
		Set<Integer> nearestNeighbors = new HashSet<Integer>();
		while (neighborsCount < searchK && !pq.isEmpty()) {
			PQEntry top = pq.poll();
			long topNodeOffset = top.nodeOffset;
			int nDescendants = getIntInAnnBuf(topNodeOffset);
			if (nDescendants == 1 && (int) (topNodeOffset / NODE_SIZE) < nItems) { // n_descendants
//				float[] v = getNodeVector(topNodeOffset);
//				if (isZeroVec(v))
//					continue;
				nearestNeighbors.add((int) (topNodeOffset / NODE_SIZE));
				neighborsCount++;
			} else if (nDescendants <= MIN_LEAF_SIZE) {
				for (int i = 0; i < nDescendants; i++) {
					int j = getIntInAnnBuf(topNodeOffset + INDEX_TYPE_OFFSET + i * INT_SIZE);
//					if (isZeroVec(getNodeVector(j * NODE_SIZE)))
//						continue;
					nearestNeighbors.add(j);
					neighborsCount++;
				}
			} else {
				float[] v = getNodeVector(topNodeOffset);
				float d = top.margin;
				float margin;
				switch (INDEX_TYPE) {
				case ANGULAR:
					margin = cosineMargin(v, queryVector, DIMENSION);
					break;
				case DOT:
					margin = dotMargin(v, queryVector, getDotFactor(topNodeOffset), DIMENSION);
					break;
				case MANHATTAN:
					margin = manhattanMargin(v, queryVector, getNodeBias(topNodeOffset), DIMENSION);
					break;
				case EUCLIDEAN:
					margin = euclideanMargin(v, queryVector, getNodeBias(topNodeOffset), DIMENSION);
					break;
				default:
					margin = 0;
					break;
				}
				long childrenMemOffset = topNodeOffset + INDEX_TYPE_OFFSET;
				long lChild = NODE_SIZE * getIntInAnnBuf(childrenMemOffset);
				long rChild = NODE_SIZE * getIntInAnnBuf(childrenMemOffset + 4);
				pq.add(new PQEntry(pqDistance(d, margin, 1), rChild));
				pq.add(new PQEntry(pqDistance(d, margin, 0), lChild));
			}
		}

		ArrayList<PQEntry> sortedNNs = new ArrayList<PQEntry>();
		for (int nn : nearestNeighbors) {
			float[] v = getItemVector(nn);
			if (!isZeroVec(v) && getIntInAnnBuf(nn * NODE_SIZE) == 1) {
				float margin;
				switch (INDEX_TYPE) {
				case ANGULAR:
					margin = angularDistance(v, queryVector, DIMENSION, getNodeBias(nn * NODE_SIZE), queryVectorDot);
					break;
				case DOT:
					margin = dot(v, queryVector, DIMENSION);
					break;
				case MANHATTAN:
					margin = manhattanDistance(v, queryVector, DIMENSION);
					break;
				case EUCLIDEAN:
					margin = euclideanDistance(v, queryVector, DIMENSION);
					break;
				default:
					margin = 0;
					break;
				}
				sortedNNs.add(new PQEntry(margin, nn));
			}
		}
		Collections.sort(sortedNNs);

		return sortedNNs.subList(0, nResults);
	}

	public static float pqDistance(float distance, float margin, int childNr) {
		if (childNr == 0) {
			margin = -margin;
		}
		return distance > margin ? margin : distance;
	}

}
