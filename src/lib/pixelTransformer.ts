import * as ort from 'onnxruntime-web';

// -----------------------------------------------
// 1. Configuration & Type Definitions
// -----------------------------------------------

const GRID_SIZE = 16;
const DIFF_RANGE = 2 * GRID_SIZE - 1; // 31

export type Coordinate = [number, number]; // [row, col]

export interface Prediction {
	coord: Coordinate;
	prob: number;
}

// -----------------------------------------------
// 2. Helper Functions
// -----------------------------------------------

/**
 * Converts a (dr, dc) diff tuple to a single integer token.
 */
function diffToToken(diff: Coordinate): number {
	const [dr, dc] = diff;
	const drShifted = dr + GRID_SIZE - 1;
	const dcShifted = dc + GRID_SIZE - 1;
	return drShifted * DIFF_RANGE + dcShifted;
}

/**
 * Converts an integer token back to a (dr, dc) diff tuple.
 */
function tokenToDiff(token: number): Coordinate {
	const drShifted = Math.floor(token / DIFF_RANGE);
	const dcShifted = token % DIFF_RANGE;
	const dr = drShifted - (GRID_SIZE - 1);
	const dc = dcShifted - (GRID_SIZE - 1);
	return [dr, dc];
}

/**
 * Applies the Softmax function to an array of numbers.
 */
function softmax(arr: Float32Array): Float32Array {
	const maxLogit = Math.max(...arr);
	const exps = arr.map((x) => Math.exp(x - maxLogit));
	const sumExps = exps.reduce((a, b) => a + b);
	return new Float32Array(exps.map((e) => e / sumExps));
}

// -----------------------------------------------
// 3. Pixel Predictor Class
// -----------------------------------------------

export class PixelPredictor {
	private session: ort.InferenceSession | null = null;

	/**
	 * Initializes the predictor by loading the ONNX model.
	 * @param modelPath - The public path to the .onnx model file.
	 */
	async init(modelPath: string): Promise<void> {
		try {
			this.session = await ort.InferenceSession.create(modelPath);
			console.log('✅ ONNX session created successfully.');
		} catch (e) {
			console.error(`Failed to create ONNX session: ${e}`);
			throw e;
		}
	}

	/**
	 * Generates a causal mask for the transformer model.
	 * @param size - The sequence length.
	 * @returns An ONNX Tensor representing the mask.
	 */
	private generateCausalMask(size: number): ort.Tensor {
		const mask = new Float32Array(size * size);
		for (let i = 0; i < size; i++) {
			for (let j = 0; j < size; j++) {
				mask[i * size + j] = j > i ? -Infinity : 0.0;
			}
		}
		return new ort.Tensor('float32', mask, [size, size]);
	}

	/**
	 * Predicts the next `k` most likely coordinates based on the input drawing.
	 * @param drawingCoords - An array of coordinates representing the user's drawing.
	 * @param k - The number of top predictions to return.
	 */
	async predict(drawingCoords: Coordinate[], k: number = 5): Promise<Prediction[]> {
		if (!this.session) {
			throw new Error('Session not initialized. Call init() first.');
		}
		if (drawingCoords.length < 2) {
			console.warn('⚠️ Drawing is too short for prediction.');
			return [];
		}

		// 1. Pre-process coordinates into model inputs
		const modelInputs: number[][] = [];
		for (let i = 0; i < drawingCoords.length - 1; i++) {
			const p1 = drawingCoords[i];
			const p2 = drawingCoords[i + 1];
			const diff: Coordinate = [p2[0] - p1[0], p2[1] - p1[1]];
			modelInputs.push([diffToToken(diff), p1[0], p1[1]]);
		}

		const seqLength = modelInputs.length;
		const flatInputs = new Int32Array(modelInputs.flat());
		const srcTensor = new ort.Tensor('int32', flatInputs, [1, seqLength, 3]);

		// 2. Generate the causal mask dynamically
		const maskTensor = this.generateCausalMask(seqLength);

		// 3. Run inference with both tensors
		const feeds = {
			src: srcTensor,
			tgt_mask: maskTensor
		};
		const results = await this.session.run(feeds);
		const logits = results.logits;

		// 4. Post-process the output
		const lastTokenLogits = logits.data.slice(
			(seqLength - 1) * logits.dims[2],
			seqLength * logits.dims[2]
		) as Float32Array;

		const probabilities = softmax(lastTokenLogits);

		const topK = Array.from(probabilities)
			.map((prob, index) => ({ prob, index }))
			.sort((a, b) => b.prob - a.prob)
			.slice(0, k);

		// 5. Format predictions
		const lastCoord = drawingCoords[drawingCoords.length - 1];
		const predictions: Prediction[] = [];

		for (const pred of topK) {
			const predDiff = tokenToDiff(pred.index);
			const nextCoord: Coordinate = [lastCoord[0] + predDiff[0], lastCoord[1] + predDiff[1]];

			if (
				nextCoord[0] >= 0 &&
				nextCoord[0] < GRID_SIZE &&
				nextCoord[1] >= 0 &&
				nextCoord[1] < GRID_SIZE
			) {
				predictions.push({
					coord: nextCoord,
					prob: pred.prob
				});
			}
		}
		return predictions;
	}
}

// -----------------------------------------------
// 4. Usage Example
// -----------------------------------------------
async function main() {
	const predictor = new PixelPredictor();

	// The model must be in a public-facing directory
	await predictor.init('./pixel_transformer.onnx');

	// An 'L' shape drawing
	const userDrawing: Coordinate[] = [
		[2, 2],
		[3, 2],
		[4, 2],
		[5, 2],
		[5, 3],
		[5, 4]
	];
	console.log(`Input drawing (coords):`, userDrawing);

	console.log(`\n🔮 Running inference...`);
	const topPredictions = await predictor.predict(userDrawing, 5);

	if (topPredictions.length > 0) {
		console.log('\n--- Top 5 Predictions ---');
		topPredictions.forEach((pred) => {
			console.log(
				`  Coordinate: [${pred.coord.join(', ')}] | Probability: ${(pred.prob * 100).toFixed(2)}%`
			);
		});
	}
}

// To run this example, ensure your HTML file includes this script as a module.
main();
