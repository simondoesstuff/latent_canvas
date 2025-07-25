import type { Coord, int } from './utils';

/**
 * Runnable ML model
 */
export interface Model {
	predict(sequence: Coord[], amnt: int): Coord[];
}
