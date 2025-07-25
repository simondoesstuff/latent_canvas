import type { Coord, int } from './utils';

export interface Model {
	predict(sequence: Coord[], amnt: int): Coord[];
}
