import tkinter as tk
from tkinter import messagebox, scrolledtext

from app import (
    run_research,
    list_saved_reports,
    read_saved_report,
)


def run_research_from_ui():
    """Validate the question, run research, and display the report in the window.

    The research button is disabled while work is in progress and restored
    afterward. Failures are shown to the user in an error dialog.
    """
    question = question_input.get("1.0", tk.END).strip()

    if not question:
        messagebox.showwarning(
            "Missing question",
            "Please enter a research question."
        )
        return

    research_button.config(state=tk.DISABLED)
    status_label.config(text="Researching...")
    root.update_idletasks()

    try:
        result = run_research(question)

        output.delete("1.0", tk.END)

        output.insert(
            tk.END,
            result["report"]
        )

        output.insert(
            tk.END,
            "\n\nSOURCES:\n"
        )

        if result["sources"]:
            for source in result["sources"]:
                if source["type"] == "web":
                    output.insert(
                        tk.END,
                        f"- {source['name']}\n"
                        f"  {source['url']}\n"
                    )
                else:
                    output.insert(
                        tk.END,
                        f"- {source['name']} "
                        "(local document)\n"
                    )
        else:
            output.insert(
                tk.END,
                "- No supporting sources were retrieved.\n"
            )

        output.insert(
            tk.END,
            f"\nSaved report:\n{result['report_path']}"
        )

        status_label.config(
            text="Research complete."
        )

    except Exception as error:
        status_label.config(
            text="Research failed."
        )

        messagebox.showerror(
            "Research error",
            str(error)
        )

    finally:
        research_button.config(state=tk.NORMAL)

def show_saved_reports():
    """Open a window for browsing saved reports and reading their contents.

    Displays an informational dialog when no saved reports are available.
    """
    reports = list_saved_reports()

    if not reports:
        messagebox.showinfo(
            "Saved Reports",
            "No saved research reports were found."
        )
        return

    reports_window = tk.Toplevel(root)
    reports_window.title("Saved Research Reports")
    reports_window.geometry("900x650")

    left_frame = tk.Frame(reports_window)
    left_frame.pack(
        side=tk.LEFT,
        fill=tk.Y,
        padx=10,
        pady=10
    )

    right_frame = tk.Frame(reports_window)
    right_frame.pack(
        side=tk.RIGHT,
        fill=tk.BOTH,
        expand=True,
        padx=10,
        pady=10
    )

    report_list = tk.Listbox(
        left_frame,
        width=45
    )
    report_list.pack(
        fill=tk.Y,
        expand=True
    )

    report_viewer = scrolledtext.ScrolledText(
        right_frame,
        wrap=tk.WORD
    )
    report_viewer.pack(
        fill=tk.BOTH,
        expand=True
    )

    for report in reports:
        report_list.insert(
            tk.END,
            report.name
        )

    def open_selected_report(event=None):
        """Load the selected report into the viewer in the reports window.

        Args:
            event: Optional Tkinter list-selection event.
        """
        selection = report_list.curselection()

        if not selection:
            return

        index = selection[0]
        report_path = reports[index]

        try:
            content = read_saved_report(
                report_path
            )

            report_viewer.delete(
                "1.0",
                tk.END
            )

            report_viewer.insert(
                tk.END,
                content
            )

        except Exception as error:
            messagebox.showerror(
                "Report error",
                str(error)
            )

    report_list.bind(
        "<<ListboxSelect>>",
        open_selected_report
    )
root = tk.Tk()
root.title("Local Research Assistant")
root.geometry("900x700")

title_label = tk.Label(
    root,
    text="Local Research Assistant",
    font=("Segoe UI", 18, "bold")
)
title_label.pack(pady=(20, 10))

instruction_label = tk.Label(
    root,
    text="Enter a research question:"
)
instruction_label.pack(anchor="w", padx=20)

question_input = scrolledtext.ScrolledText(
    root,
    height=5,
    wrap=tk.WORD
)
question_input.pack(
    fill=tk.X,
    padx=20,
    pady=(5, 10)
)

research_button = tk.Button(
    root,
    text="Research",
    command=run_research_from_ui
)
research_button.pack(pady=5)

saved_reports_button = tk.Button(
    root,
    text="Saved Reports",
    command=show_saved_reports
)

saved_reports_button.pack(pady=5)

status_label = tk.Label(
    root,
    text="Ready"
)
status_label.pack(pady=5)

output_label = tk.Label(
    root,
    text="Research Report"
)
output_label.pack(
    anchor="w",
    padx=20,
    pady=(10, 0)
)

output = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD
)
output.pack(
    fill=tk.BOTH,
    expand=True,
    padx=20,
    pady=(5, 20)
)

root.mainloop()