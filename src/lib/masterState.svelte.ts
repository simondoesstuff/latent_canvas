import { goto } from '$app/navigation';
import { page } from '$app/state';
import { onMount } from 'svelte';
import { GridStack } from './gridStack.svelte';
import type { Model } from './model';
import { Momento } from './momento';
import { PixelTransformerAdapter } from './pixelTransformerAdapter';
import type { Coord } from './utils';

/**
 * The master state controls the state for the overall system.
 * The "model" in MVC
 */
export class MasterState {
	private model: Model = $state(new PixelTransformerAdapter());
	private predictions: Coord[] = $state([]);
	private grid: GridStack;

	public constructor(grid: GridStack) {
		grid ??= new GridStack([16, 16]);
		this.grid = $state(grid);

		// initialize a random grid state
		for (let y = 0; y < 16; y++) {
			for (let x = 0; x < 16; x++) {
				if (Math.random() > 0.9) {
					this.grid.push(x, y);
				}
			}
		}

		// this handles the URL flag state

		onMount(() => {
			const urlState = page.url.searchParams.get('grid');
			if (!urlState) return;
			const momento = Momento.fromStr(urlState);
			if (!momento) return;
			this.getGrid().restoreMomento(momento);
			this.refreshPredictions();
		});
	}

	private updateURL() {
		const serializedState = Momento.fromDiffs(this.grid.getCoordSeq()).asString();
		page.url.searchParams.set('grid', serializedState);
		goto(page.url, { replaceState: true, keepFocus: true });
	}

	private refreshPredictions() {
		this.predictions = this.model.predict(this.grid.getCoordSeq(), 3);
	}

	public draw(x: number, y: number) {
		if (this.grid.at(x, y)) return;
		this.grid.push(x, y);
		this.refreshPredictions();
		this.updateURL();
	}

	public clearState() {
		this.grid.clear();
		this.predictions = [];
		this.updateURL();
	}

	public undo() {
		this.grid.pop();
		this.refreshPredictions();
		this.updateURL();
	}

	public getPredictions() {
		return this.predictions;
	}

	public getGrid() {
		return this.grid;
	}
}
