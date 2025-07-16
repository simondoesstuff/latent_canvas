<script lang="ts">
	import Grid from './Grid.svelte';

	const state: boolean[][] = $state([]);

	for (let y = 0; y < 16; y++) {
		let row: boolean[] = [];

		for (let x = 0; x < 16; x++) {
			const v = Math.random() > 0.8;
			row.push(v);
		}

		state.push(row);
	}

	function onDraw(x: number, y: number) {
		state[y][x] = true;
	}
</script>

<div class="container mx-auto grid h-dvh w-dvw place-items-center p-4">
	<Grid {state} ondraw={onDraw} />
</div>
