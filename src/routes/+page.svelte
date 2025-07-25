<script lang="ts">
	import Close from '$lib/icons/Close.svelte';
	import Grid from './Grid.svelte';
	import Trash from '$lib/icons/Trash.svelte';
	import Undo from '$lib/icons/Undo.svelte';
	import { cubicOut } from 'svelte/easing';
	import { scale } from 'svelte/transition';
	import { GridStack, type Coord } from '$lib/gridStack.svelte.ts';
	import { PixelTransformerAdapter } from '$lib/pixelTransformerAdapter.ts';
	import { MasterState } from '$lib/masterState.svelte.ts';
	import { onMount } from 'svelte';
	import { page } from '$app/state';
	import { Momento } from '$lib/momento';

	// this is the primary state of the system

	let state = $state(new MasterState(new GridStack([16, 16])));

	// this is svelte's way of establishing the observer pattern

	let grid: GridStack = $derived(state.getGrid());
	let predictions: Coord[] = $derived(state.getPredictions());

	// modal local state

	let isModalOpen: boolean = $state(false);
	const openModal = () => (isModalOpen = true);
	const closeModal = () => (isModalOpen = false);

	function onTrash() {
		openModal();
	}

	// callbacks

	function onDraw(x: number, y: number) {
		state.draw(x, y);
	}

	function clearState() {
		state.clearState();
	}

	function onUndo() {
		state.undo();
	}
</script>

<!-- CtrlZ Hook -->
<svelte:window
	onkeypress={(e) => {
		if (e.ctrlKey && e.key.toLowerCase() === 'z') {
			e.preventDefault();
			onUndo();
		}
	}}
/>

<!-- Modal -->
{#if isModalOpen}
	<div class="big grid-center absolute z-10">
		<div
			transition:scale={{ start: 0.7, opacity: 0, duration: 210, easing: cubicOut }}
			class="bg-bg border-fg grid-center relative z-10 border-2 p-5"
		>
			<h1 class="prose-2xl mb-3">Are you sure you want to do that?</h1>
			<button
				onclick={() => {
					clearState();
					closeModal();
				}}
				class="hover:text-bg m-3 rounded bg-red-400 p-5"
			>
				Erase the board
			</button>
			<button
				aria-label="close modal"
				onclick={closeModal}
				class="bg-fg text-bg absolute top-0 right-0 aspect-square w-6 translate-x-[50%] -translate-y-[50%] rounded-full"
			>
				<Close />
			</button>
		</div>
		<div onclick={closeModal} class="big absolute bg-black opacity-60"></div>
	</div>
{/if}

<!-- main -->
<main>
	<div class="grid-center h-dvh w-dvw">
		<div
			class="flex max-md:h-full max-md:flex-col max-md:justify-center md:w-full md:justify-around"
		>
			<button onclick={onTrash} class="max-md:hidden"><Trash /></button>
			<Grid state={grid} {predictions} ondraw={onDraw} />
			<button onclick={onUndo} class="max-md:hidden"><Undo /></button>
			<!-- max-md: Under buttons -->
			<div class="my-8 flex w-full justify-center gap-5 md:hidden">
				<button onclick={onTrash}><Trash /></button>
				<button onclick={onUndo}><Undo /></button>
			</div>
		</div>
	</div>
</main>

<style lang="postcss">
	@reference "../app.css";

	button {
		@apply cursor-pointer transition-all hover:scale-115 hover:brightness-155 active:scale-90;
	}

	main button {
		@apply bg-bg-2 my-auto h-16 min-w-16 rounded-4xl p-2;
	}
</style>
