<script lang="ts">
	import type { GridStack } from '$lib/gridStack';
	import type { Coord } from '$lib/utils';
	import { fade } from 'svelte/transition';

	let {
		state,
		predictions,
		ondraw
	}: { state: GridStack; predictions: Coord[]; ondraw: (x: number, y: number) => void } = $props();

	// this converts the predictions into something compatible for the UI
	const predictionsSet = $derived(new Set(predictions.map((x) => x[1] * 16 + x[0])));

	function doesModelPredict(x, y) {
		return predictionsSet.has(y * 16 + x);
	}

	// this state manages the cursor moving in and out of cells

	let drawing = false;

	function onMouseDown(x: number, y: number) {
		drawing = true;
		ondraw(x, y);
	}

	function onMouseUp() {
		drawing = false;
	}

	function onMouseEnter(x: number, y: number) {
		if (!drawing) return;
		ondraw(x, y);
	}
</script>

<svelte:window onmouseup={onMouseUp} onmouseleave={onMouseUp} />

<div class="flex flex-col">
	{#each state as row, y}
		<div class="flex">
			{#each row as item, x}
				<!-- Tile Element -->
				{#key item}
					<button
						aria-label="tile"
						class:bg-primary={doesModelPredict(x, y)}
						class:bg-primary-3={item && !doesModelPredict(x, y)}
						class:bg-primary-4={!item && !doesModelPredict(x, y)}
						onmousedown={() => onMouseDown(x, y)}
						onmouseenter={() => onMouseEnter(x, y)}
						in:fade={{ duration: 150 }}
						class="hover:bg-primary border-bg aspect-square w-8 cursor-pointer border-[0.15rem] hover:scale-125 hover:border-0 hover:brightness-115"
					></button>
				{/key}
			{/each}
		</div>
	{/each}
</div>
