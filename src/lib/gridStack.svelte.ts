import { Momento } from './momento';
import type { Coord, int } from './utils';

export class GridStack {
	private _grid: boolean[][] = $state([]);
	private _diffs: Coord[] = $state([]);
	// in TS, fields can be made public without violating
	// encapsulation because getters/setters can be added
	// later without external refactor
	public readonly shape: Coord;

	// the constructor in TS
	public constructor(shape: Coord) {
		this.shape = shape;
		// generate a 2D matrix of proper shape initialized false
		this._grid = Array(shape[1])
			.fill(0)
			.map(() => Array(shape[0]).fill(false));
	}

	// specify a tile as enabled and push the diff on to the stack
	public push(x: int, y: int) {
		if (this._grid[y][x]) throw Error('Cells already enabled cannot be re-drawn.');

		this._grid[y][x] = true;
		this._diffs.push([x, y]);
	}

	// pop a diff off the stack, updating the grid accordingly
	public pop(): Coord | undefined {
		const diff = this._diffs.pop();

		if (diff != null) {
			const [x, y] = diff;
			this._grid[y][x] = false;
		}

		return diff;
	}

	public clear() {
		// for all, cell -> false
		this._grid = this._grid.map((row) => row.map(() => false));
		this._diffs = [];
	}

	// get the value at a grid tile
	public at(x: int, y: int): boolean {
		return this._grid[y][x];
	}

	// This makes the class iterable
	*[Symbol.iterator](): Iterator<Array<boolean>> {
		for (let y = 0; y < this.shape[1]; y++) {
			yield [...this._grid[y]];
		}
	}

	public getCoordSeq(): Array<Coord> {
		return [...this._diffs];
	}

	// this is where the momento pattern is implemented

	public restoreMomento(momento: Momento) {
		const diffs = momento.asDiffs();
		this.clear();

		for (const diff of diffs) {
			this.push(...diff);
		}
	}

	public createMomento() {
		return Momento.fromDiffs(this._diffs);
	}
}
