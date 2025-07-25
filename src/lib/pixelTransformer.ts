/**
 * The placeholder for the model class which would have made a call to a compiled
 * deep-learning model using ONNX.
 */
export class PixelTransformer {
	public call(inputVector: number[][]): number[] {
		// TODO: I could not successfully pair the model into the front end. So I am implementing a placeholder. See the assignment summary.
		return new Array(inputVector[0].length).fill(0).map(() => Math.random());
	}
}
