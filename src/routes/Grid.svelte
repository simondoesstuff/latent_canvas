<script lang="ts">
	import { fade } from 'svelte/transition';

	let { state, ondraw }: { state: boolean[][]; ondraw: (x: number, y: number) => void } = $props();

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
						class:bg-primary-2={item}
						class:bg-primary-3={!item}
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
