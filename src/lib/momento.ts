import type { GridStack } from './gridStack.svelte';
import type { Coord, int } from './utils';

/**
 * This class serves as the momento for GridStack.
 * It implements the de/serialize pattern with a basic factory for safety
 */
export class Momento {
	private dense: string;
	private diffs: Coord[];

	private constructor(dense: string, diffs: Coord[]) {
		this.dense = dense;
		this.diffs = diffs;
	}

	/**
	 * Diffs are serialized (into a string form) in subsequent calls.
	 * This is primarily used to create the Momento from existing objects.
	 */
	public static fromDiffs(diffs: Coord[]) {
		const dense = Momento.serialize(diffs);
		return new Momento(dense, diffs);
	}

	/**
	 * Parses the string to build the Momento
	 * The string must be in the expected format -- or risk undefined behavior
	 */
	public static fromStr(dense: string) {
		const diffs = Momento.deserialize(dense);
		return new Momento(dense, diffs);
	}

	private static serialize(diffs: Coord[]): string {
		const ids = diffs.map((x) => x[1] * 16 + x[0]);
		const json = JSON.stringify(ids);
		return encodeURIComponent(json);
	}

	private static deserialize(expr: string): Coord[] {
		const json = decodeURIComponent(expr);
		const ids: int[] = JSON.parse(json);
		const diffs: Coord[] = ids.map((x) => [x % 16, Math.floor(x / 16)] as Coord);
		return diffs;
	}

	public asString() {
		return this.dense;
	}

	public asDiffs() {
		return this.diffs;
	}
}
