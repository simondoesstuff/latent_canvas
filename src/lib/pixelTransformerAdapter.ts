import { PixelTransformer } from './pixelTransformer';
import type { Coord, int } from './utils';

/**
 * An adapter layer around the model to make it easier to use.
 * The model itself uses one-hot encoded vectors as IO.
 */
export class PixelTransformerAdapter {
	private model: PixelTransformer;

	public constructor() {
		this.model = new PixelTransformer();
	}

	private encode(coord: Coord): number[] {
		const [x, y] = coord;
		// encode as an input vector
		const index = y * 16 + x;
		const vector = Array(256).fill(0);
		vector[index] = 1;
		return vector;
	}

	private predictSingle(sequence: Coord[]): Coord {
		// encode the sequence
		const inputVector = sequence.map((x) => this.encode(x));
		// call the model
		const outputVector = this.model.call(inputVector);
		// decode into a prediction, taking the best scoring one
		const max = Math.max(...outputVector);
		const outIndex = outputVector.indexOf(max); // inefficient argmax
		return [outIndex % 16, Math.floor(outIndex / 16)];
	}

	/**
	 * Predict a series of coordinates the follow from the sequence
	 */
	public predict(sequence: Coord[], amnt: int = 3): Coord[] {
		const runningSeq = sequence;
		const output: Coord[] = [];

		for (let i = 0; i < amnt; i++) {
			const pred = this.predictSingle(runningSeq);
			output.push(pred);
			runningSeq.push(pred);
		}

		return output;
	}
}
